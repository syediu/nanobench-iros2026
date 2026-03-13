"""BC-MLP baseline using HumanCompatibleAI/imitation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class BCMLPConfig:
    seed: int = 42
    batch_size: int = 512
    lr: float = 1e-3
    n_epochs: int = 100


def train_bc_mlp(
    transitions,
    obs_dim: int = 14,
    act_dim: int = 4,
    cfg: Optional[BCMLPConfig] = None,
    device: str = "cpu",
):
    """
    Train a behavior cloning MLP policy with imitation.algorithms.bc.BC.
    Returns (bc_trainer, policy).
    """
    if cfg is None:
        cfg = BCMLPConfig()

    try:
        import gymnasium as gym
        from imitation.algorithms import bc
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Missing dependencies for BC-MLP. Install `gymnasium` and `imitation`.\n"
            f"Original error: {e}"
        )

    rng = np.random.default_rng(cfg.seed)
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=0.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    bc_trainer = bc.BC(
        observation_space=obs_space,
        action_space=act_space,
        demonstrations=transitions,
        batch_size=cfg.batch_size,
        optimizer_kwargs={"lr": cfg.lr},
        rng=rng,
        device=device,
    )
    bc_trainer.train(n_epochs=cfg.n_epochs)
    return bc_trainer, bc_trainer.policy


class BCMLPController:
    """Wrap an imitation policy into the Task 2 ControllerInterface."""

    def __init__(self, policy: torch.nn.Module, device: torch.device):
        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        x = torch.from_numpy(obs).to(self.device)
        a = self.policy(x)
        if isinstance(a, tuple):
            a = a[0]
        a_np = a.detach().cpu().numpy().reshape(-1)
        return np.clip(a_np, 0.0, 1.0)


def save_policy(policy: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), path)


def load_policy(policy_ctor, path: str | Path, device: torch.device) -> torch.nn.Module:
    """
    Load policy weights into a provided constructor (so we don't depend on pickles).
    policy_ctor: callable returning an uninitialized torch.nn.Module with same structure.
    """
    path = Path(path)
    policy = policy_ctor().to(device)
    state = torch.load(path, map_location=device)
    policy.load_state_dict(state)
    policy.eval()
    return policy

