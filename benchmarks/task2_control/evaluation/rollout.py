"""Closed-loop rollout engine for Task 2 (learned dynamics plant).

Key stability features:
- scaler sanity check (detect mismatch early)
- physical clipping + quaternion renormalization
- divergence detection + early termination with NaN padding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class ControllerInterface(Protocol):
    """Common controller interface used by the rollout engine."""

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """obs: (obs_dim,) normalized -> action: (4,) normalized in [0,1]."""


@dataclass(frozen=True)
class RolloutResult:
    rollout_obs: np.ndarray
    gt_obs: np.ndarray
    actions_pred: np.ndarray
    actions_expert: np.ndarray
    valid_steps: int
    diverged: bool
    divergence_step: Optional[int]


def clip_state_to_physical_bounds(state: np.ndarray) -> np.ndarray:
    """
    Clip rollout state to physically plausible bounds and renormalize quaternion.

    state layout:
      [pos(3), vel(3), quat_wxyz(4), setpoint(3), vbat(1)]
    """
    s = np.asarray(state, dtype=np.float32).copy()
    # Arena ~4m; keep conservative bounds
    s[0:3] = np.clip(s[0:3], -5.0, 5.0)
    s[3:6] = np.clip(s[3:6], -5.0, 5.0)
    s[6:10] = np.clip(s[6:10], -1.0, 1.0)
    q = s[6:10]
    n = float(np.linalg.norm(q))
    if np.isfinite(n) and n > 1e-6:
        s[6:10] = q / n
    return s


@torch.no_grad()
def rollout_controller(
    controller: ControllerInterface,
    dynamics_model: torch.nn.Module,
    test_obs: np.ndarray,
    test_actions_expert: np.ndarray,
    obs_scaler: StandardScaler,
    device: torch.device,
    override_exogenous: bool = True,
    dynamics_predicts_delta: bool = False,
    divergence_threshold_m: float = 2.0,
) -> RolloutResult:
    """
    Roll out a controller closed-loop through a learned dynamics model.

    - test_obs: (T,14) unnormalized obs sequence from aligned.csv
    - test_actions_expert: (T-1,4) expert motor actions in [0,1] (for action_MSE)
    - dynamics_model: expects (obs_scaled, act)-> next_obs_scaled (batchwise)
    - override_exogenous: if True, replace [setpoint,vbat] in predicted obs
      with the logged values at the next timestep (t+1). This keeps exogenous
      signals realistic and avoids "predicting" the command generator.
    """
    gt = np.asarray(test_obs, dtype=np.float32)
    T = int(gt.shape[0])
    if T < 2:
        raise ValueError("test_obs too short.")

    rollout = np.zeros_like(gt, dtype=np.float32)
    rollout[0] = gt[0]

    u_pred = np.zeros((T - 1, 4), dtype=np.float32)
    u_exp = np.asarray(test_actions_expert, dtype=np.float32)
    if u_exp.shape[0] != T - 1:
        raise ValueError("test_actions_expert must have length T-1.")

    dynamics_model = dynamics_model.to(device)
    dynamics_model.eval()

    # Scaler sanity check on initial state (detect mismatch early)
    x0n = obs_scaler.transform(rollout[0:1]).astype(np.float32)
    if not np.all(np.isfinite(x0n)) or np.any(np.abs(x0n) > 10.0):
        raise ValueError(
            f"Scaler mismatch: normalized initial state has extreme values: {x0n}."
        )

    valid_steps = T
    diverged = False
    divergence_step: Optional[int] = None

    for t in range(T - 1):
        obs_scaled = obs_scaler.transform(rollout[t : t + 1]).astype(np.float32)
        a = controller.predict(obs_scaled[0])
        a = np.asarray(a, dtype=np.float32).reshape(4,)
        a = np.clip(a, 0.0, 1.0)
        u_pred[t] = a

        obs_t = torch.from_numpy(obs_scaled).to(device)
        act_t = torch.from_numpy(a[None, :]).to(device)

        pred = dynamics_model(obs_t, act_t).detach().cpu().numpy().astype(np.float32)
        next_scaled = (obs_scaled + pred) if dynamics_predicts_delta else pred
        next_obs = obs_scaler.inverse_transform(next_scaled).astype(np.float32)
        rollout[t + 1] = clip_state_to_physical_bounds(next_obs[0])

        if override_exogenous:
            # setpoint (10:13) and vbat (13) are treated as exogenous
            rollout[t + 1, 10:13] = gt[t + 1, 10:13]
            rollout[t + 1, 13] = gt[t + 1, 13]

        # Divergence detection (position error vs GT)
        pos_err = float(np.linalg.norm(rollout[t + 1, 0:3] - gt[t + 1, 0:3]))
        if not np.isfinite(pos_err) or pos_err > divergence_threshold_m:
            diverged = True
            divergence_step = t + 1
            valid_steps = divergence_step
            # pad remainder with NaN to keep metrics well-defined
            rollout[divergence_step:] = np.nan
            u_pred[t + 1 :] = np.nan
            break

    return RolloutResult(
        rollout_obs=rollout,
        gt_obs=gt,
        actions_pred=u_pred,
        actions_expert=u_exp,
        valid_steps=valid_steps,
        diverged=diverged,
        divergence_step=divergence_step,
    )

