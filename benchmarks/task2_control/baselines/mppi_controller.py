"""MPPI baseline using UM-ARM-Lab/pytorch_mppi.

This controller plans motor commands using the learned dynamics model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch

try:
    from pytorch_mppi import MPPI
except Exception as e:  # pragma: no cover
    MPPI = None  # type: ignore
    _MPPI_IMPORT_ERROR = e


@dataclass(frozen=True)
class MPPIConfig:
    seed: int = 42
    horizon: int = 10
    num_samples: int = 128
    lambda_: float = 0.1
    noise_sigma: float = 0.05  # std dev per action dim in normalized units


def _ensure_mppi_available() -> None:
    if MPPI is None:  # pragma: no cover
        raise ImportError(
            "Failed to import pytorch_mppi. Install with `pip install pytorch-mppi`.\n"
            f"Original error: {_MPPI_IMPORT_ERROR}"
        )


def make_running_cost_from_reference(
    ref_obs_scaled: np.ndarray,
    obs_mean: np.ndarray,
    obs_scale: np.ndarray,
    action_weight: float = 0.01,
) -> Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], list]:
    """
    Running cost for MPPI:
      pos tracking cost to reference setpoint position (from ref_obs_scaled[:,10:13])
      + small action penalty.

    state is in *scaled obs space*; we unscale pos/setpoint using mean/scale.
    """
    ref = np.asarray(ref_obs_scaled, dtype=np.float32)
    mean = torch.as_tensor(obs_mean, dtype=torch.float32)
    scale = torch.as_tensor(obs_scale, dtype=torch.float32)
    step_counter = [0]

    def running_cost(state_scaled: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        t = step_counter[0]
        t = int(np.clip(t, 0, len(ref) - 1))
        ref_t = torch.as_tensor(ref[t], dtype=torch.float32, device=state_scaled.device)

        # unscale positions
        pos = state_scaled[:, 0:3] * scale[0:3].to(state_scaled.device) + mean[0:3].to(state_scaled.device)
        sp = ref_t[10:13] * scale[10:13].to(state_scaled.device) + mean[10:13].to(state_scaled.device)

        pos_cost = torch.sum((pos - sp) ** 2, dim=-1)
        act_cost = action_weight * torch.sum(action ** 2, dim=-1)
        return pos_cost + act_cost

    return running_cost, step_counter


def make_running_cost_from_state_setpoint(
    obs_mean: np.ndarray,
    obs_scale: np.ndarray,
    pos_weight: float = 1.0,
    vel_weight: float = 0.05,
    action_weight: float = 0.01,
) -> Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], list]:
    """
    Running cost that does NOT require an external time-indexed reference.

    Uses setpoint embedded in the observation itself:
      cost = pos_weight * ||pos - sp||^2 + vel_weight * ||vel||^2 + action_weight * ||u||^2

    This avoids the common pitfall where the reference index does not advance
    correctly inside MPPI's internal horizon simulation.
    """
    mean = torch.as_tensor(obs_mean, dtype=torch.float32)
    scale = torch.as_tensor(obs_scale, dtype=torch.float32)
    step_counter = [0]  # kept for API symmetry; not used by this cost

    def running_cost(state_scaled: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dev = state_scaled.device
        m = mean.to(dev)
        s = scale.to(dev)

        pos = state_scaled[:, 0:3] * s[0:3] + m[0:3]
        vel = state_scaled[:, 3:6] * s[3:6] + m[3:6]
        sp = state_scaled[:, 10:13] * s[10:13] + m[10:13]

        pos_cost = pos_weight * torch.sum((pos - sp) ** 2, dim=-1)
        vel_cost = vel_weight * torch.sum(vel**2, dim=-1)
        act_cost = action_weight * torch.sum(action**2, dim=-1)
        return pos_cost + vel_cost + act_cost

    return running_cost, step_counter


def make_running_cost_with_bc_prior(
    obs_mean: np.ndarray,
    obs_scale: np.ndarray,
    bc_controller,
    pos_weight: float = 1.0,
    vel_weight: float = 0.05,
    action_weight: float = 0.01,
    prior_weight: float = 1.0,
) -> Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], list]:
    """
    Running cost that adds an action prior around a baseline BC policy.

    cost = pos_weight * ||pos - sp||^2
           + vel_weight * ||vel||^2
           + action_weight * ||u||^2
           + prior_weight * ||u - u_BC(x)||^2

    Here u_BC(x) is given by the provided `bc_controller.predict` operating on
    *normalized* observations (same space as other controllers).
    """
    mean = torch.as_tensor(obs_mean, dtype=torch.float32)
    scale = torch.as_tensor(obs_scale, dtype=torch.float32)
    step_counter = [0]  # kept for API symmetry

    def running_cost(state_scaled: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dev = state_scaled.device
        m = mean.to(dev)
        s = scale.to(dev)

        # Unscale core state components
        pos = state_scaled[:, 0:3] * s[0:3] + m[0:3]
        vel = state_scaled[:, 3:6] * s[3:6] + m[3:6]
        sp = state_scaled[:, 10:13] * s[10:13] + m[10:13]

        pos_cost = pos_weight * torch.sum((pos - sp) ** 2, dim=-1)
        vel_cost = vel_weight * torch.sum(vel**2, dim=-1)

        # Compute BC prior action for each sample (Python loop over batch is OK for K~500)
        state_np = state_scaled.detach().cpu().numpy()
        u_bc_list = [bc_controller.predict(s_i) for s_i in state_np]
        u_bc = torch.as_tensor(np.asarray(u_bc_list, dtype=np.float32), device=dev)

        prior_cost = prior_weight * torch.sum((action - u_bc) ** 2, dim=-1)
        act_cost = action_weight * torch.sum(action**2, dim=-1)
        return pos_cost + vel_cost + act_cost + prior_cost

    return running_cost, step_counter


class MPPIController:
    """MPPI wrapper implementing ControllerInterface (predict on normalized obs)."""

    def __init__(
        self,
        dynamics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        running_cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        step_counter: list,
        obs_dim: int,
        cfg: MPPIConfig,
        device: torch.device,
    ):
        _ensure_mppi_available()
        self.device = device
        self.step_counter = step_counter

        noise_sigma = torch.diag(torch.ones(4, device=device) * float(cfg.noise_sigma) ** 2)

        self.ctrl = MPPI(
            dynamics=dynamics_fn,
            running_cost=running_cost_fn,
            nx=obs_dim,
            noise_sigma=noise_sigma,
            num_samples=cfg.num_samples,
            horizon=cfg.horizon,
            u_min=torch.zeros(4, device=device),
            u_max=torch.ones(4, device=device),
            lambda_=cfg.lambda_,
            device=device,
        )

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        # MPPI expects (nx,) state; we provide normalized obs
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        u = self.ctrl.command(x)
        # advance time for the running cost closure
        self.step_counter[0] += 1
        return np.clip(u.detach().cpu().numpy().reshape(-1), 0.0, 1.0)

