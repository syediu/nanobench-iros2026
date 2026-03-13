"""Learned dynamics model for Task 2 closed-loop rollouts.

We train a simple MLP dynamics model on (obs_t_scaled, act_t) -> obs_{t+1}_scaled.

This is intentionally separate from the SysID project; if a Task 1 dynamics model
is available, you can wrap it here later, but the default path is self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class DynamicsMLPConfig:
    seed: int = 42
    hidden_sizes: Tuple[int, int, int] = (256, 256, 256)
    lr: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 200
    patience: int = 20
    predict_delta: bool = True
    action_noise_std: float = 0.04  # Gaussian noise on actions during training (normalized [0,1]) for robustness to BC/MPPI distribution shift


class MLPDynamics(nn.Module):
    """MLP dynamics: (obs_scaled, act) -> delta_obs_scaled (preferred) or next_obs_scaled."""

    def __init__(self, obs_dim: int = 14, act_dim: int = 4, hidden: Tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        layers: List[nn.Module] = []
        d = obs_dim + act_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        layers.append(nn.Linear(d, obs_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_scaled: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_scaled, act], dim=-1)
        return self.net(x)


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_dynamics_mlp(
    transitions_train,
    transitions_val,
    cfg: Optional[DynamicsMLPConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[MLPDynamics, Dict[str, List[float]]]:
    """
    Train MLP dynamics on scaled observations.

    transitions_* are imitation Transitions objects with:
      obs, acts, next_obs
    """
    if cfg is None:
        cfg = DynamicsMLPConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _set_seeds(cfg.seed)

    model = MLPDynamics(hidden=cfg.hidden_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    loss_fn = nn.MSELoss()

    # Ensure contiguous writable arrays before torch conversion (avoids non-writable warnings)
    Xtr_np = np.array(transitions_train.obs, copy=True)
    Utr_np = np.array(transitions_train.acts, copy=True)
    Ytr_np = np.array(transitions_train.next_obs, copy=True)
    Xva_np = np.array(transitions_val.obs, copy=True)
    Uva_np = np.array(transitions_val.acts, copy=True)
    Yva_np = np.array(transitions_val.next_obs, copy=True)

    Xtr = torch.from_numpy(Xtr_np).to(device, dtype=torch.float32)
    Utr = torch.from_numpy(Utr_np).to(device, dtype=torch.float32)
    if cfg.predict_delta:
        # delta in scaled space; do not learn exogenous components (setpoint,vbat)
        Ytr_np = Ytr_np - Xtr_np
        Ytr_np[:, 10:14] = 0.0
        Yva_np = Yva_np - Xva_np
        Yva_np[:, 10:14] = 0.0

    Ytr = torch.from_numpy(Ytr_np).to(device, dtype=torch.float32)

    Xva = torch.from_numpy(Xva_np).to(device, dtype=torch.float32)
    Uva = torch.from_numpy(Uva_np).to(device, dtype=torch.float32)
    Yva = torch.from_numpy(Yva_np).to(device, dtype=torch.float32)

    n = Xtr.shape[0]
    idx = torch.arange(n, device=device)

    best_val = float("inf")
    best_state = None
    bad = 0
    hist = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.max_epochs):
        model.train()
        perm = idx[torch.randperm(n, device=device)]
        losses = []
        for i in range(0, n, cfg.batch_size):
            j = perm[i : i + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            U_batch = Utr[j]
            if cfg.action_noise_std > 0:
                U_batch = U_batch + torch.randn_like(U_batch, device=device) * cfg.action_noise_std
                U_batch = torch.clamp(U_batch, 0.0, 1.0)
            yhat = model(Xtr[j], U_batch)
            loss = loss_fn(yhat, Ytr[j])
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            yhat = model(Xva, Uva)
            vloss = float(loss_fn(yhat, Yva).item())

        tloss = float(np.mean(losses)) if losses else float("nan")
        hist["train_loss"].append(tloss)
        hist["val_loss"].append(vloss)

        if np.isfinite(vloss) and vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        sched.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = float(opt.param_groups[0]["lr"])
            print(f"[Dyn-MLP] epoch {epoch+1:04d} lr={lr:.2e} train={tloss:.6f} val={vloss:.6f}")

        if bad >= cfg.patience and best_state is not None:
            print(f"[Dyn-MLP] Early stopping at epoch {epoch+1} (best val={best_val:.6f}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, hist


def save_dynamics(model: nn.Module, path: str | Path, extra: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model_state": model.state_dict(), "extra": extra or {}}
    torch.save(ckpt, path)


def load_dynamics(path: str | Path, device: torch.device) -> Tuple[MLPDynamics, Dict[str, object]]:
    ckpt = torch.load(Path(path), map_location=device)
    model = MLPDynamics().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    extra = ckpt.get("extra", {}) if isinstance(ckpt, dict) else {}
    return model, extra

