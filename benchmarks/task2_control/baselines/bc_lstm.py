"""BC-LSTM baseline (custom PyTorch training loop)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BCLSTMConfig:
    seed: int = 42
    seq_len: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    max_epochs: int = 200
    patience: int = 20
    grad_clip: float = 1.0
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1


class LSTMController(nn.Module):
    """
    Sequence-to-action controller.

    Inputs are normalized observations: (B, T, 14).
    Outputs are normalized motor commands in [0,1]: (B, 4).
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class SlidingWindowDataset(Dataset):
    """Sliding windows within each trajectory (never across file boundaries)."""

    def __init__(self, obs_list: List[np.ndarray], act_list: List[np.ndarray], seq_len: int):
        self.seq_len = int(seq_len)
        self._index: List[Tuple[int, int]] = []
        self.obs_list = obs_list
        self.act_list = act_list

        for traj_i, (obs, act) in enumerate(zip(obs_list, act_list)):
            n = len(obs)
            if n < self.seq_len:
                continue
            # predict action at the last timestep of window
            for start in range(0, n - self.seq_len + 1):
                end = start + self.seq_len
                self._index.append((traj_i, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        traj_i, start = self._index[idx]
        end = start + self.seq_len
        x = self.obs_list[traj_i][start:end]  # (T,14)
        y = self.act_list[traj_i][end - 1]    # (4,)
        return torch.from_numpy(x), torch.from_numpy(y)


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_bc_lstm(
    train_obs_list: List[np.ndarray],
    train_act_list: List[np.ndarray],
    val_obs_list: List[np.ndarray],
    val_act_list: List[np.ndarray],
    cfg: Optional[BCLSTMConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[LSTMController, Dict[str, List[float]]]:
    """
    Train an LSTM policy from normalized obs sequences to normalized motor commands.
    Returns (best_model, history).
    """
    if cfg is None:
        cfg = BCLSTMConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _set_seeds(cfg.seed)

    model = LSTMController(
        input_dim=14,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        output_dim=4,
        dropout=cfg.dropout,
    ).to(device)

    print(f"[BC-LSTM] Trainable parameters: {_count_params(model):,}")

    train_ds = SlidingWindowDataset(train_obs_list, train_act_list, cfg.seq_len)
    val_ds = SlidingWindowDataset(val_obs_list, val_act_list, cfg.seq_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    hist = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.max_epochs):
        model.train()
        tr_losses = []
        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
                yhat = model(x)
                loss = loss_fn(yhat, y)
                va_losses.append(float(loss.item()))

        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va = float(np.mean(va_losses)) if va_losses else float("nan")
        hist["train_loss"].append(tr)
        hist["val_loss"].append(va)

        if np.isfinite(va) and va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        sched.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = float(opt.param_groups[0]["lr"])
            print(f"[BC-LSTM] epoch {epoch+1:04d}  lr={lr:.2e}  train={tr:.5f}  val={va:.5f}")

        if bad_epochs >= cfg.patience and best_state is not None:
            print(f"[BC-LSTM] Early stopping at epoch {epoch+1} (best val={best_val:.5f}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, hist


class BCLSTMController:
    """Wrap trained LSTM for ControllerInterface."""

    def __init__(self, model: LSTMController, seq_len: int, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.seq_len = int(seq_len)
        self.device = device
        self._buf: List[np.ndarray] = []

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._buf.append(obs)
        if len(self._buf) > self.seq_len:
            self._buf = self._buf[-self.seq_len :]
        # pad at beginning with the first observation (deterministic)
        if len(self._buf) < self.seq_len:
            pad = [self._buf[0]] * (self.seq_len - len(self._buf))
            seq = np.stack(pad + self._buf, axis=0)
        else:
            seq = np.stack(self._buf, axis=0)
        x = torch.from_numpy(seq[None, :, :]).to(self.device, dtype=torch.float32)
        a = self.model(x).detach().cpu().numpy().reshape(-1)
        return np.clip(a, 0.0, 1.0)


def save_checkpoint(model: nn.Module, path: str | Path, extra: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model_state": model.state_dict()}
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(Path(path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

