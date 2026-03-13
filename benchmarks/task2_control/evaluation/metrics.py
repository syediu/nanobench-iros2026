"""Metrics for Task 2 controller benchmarking (offline + closed-loop rollouts)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class BatteryBin:
    name: str
    lo: float
    hi: float

    def mask(self, vbat: np.ndarray) -> np.ndarray:
        v = np.asarray(vbat, dtype=float)
        return (v >= self.lo) & (v < self.hi)


BATTERY_BINS = [
    BatteryBin("fresh", 3.9, 10.0),
    BatteryBin("mid", 3.7, 3.9),
    BatteryBin("depleted", -10.0, 3.7),
]


def rmse(vec: np.ndarray) -> float:
    vec = np.asarray(vec, dtype=float)
    return float(np.sqrt(np.mean(vec * vec)))


def pos_errors(pos_roll: np.ndarray, pos_gt: np.ndarray) -> np.ndarray:
    e = np.asarray(pos_roll, dtype=float) - np.asarray(pos_gt, dtype=float)
    return np.linalg.norm(e, axis=1)


def yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Yaw from quaternion in (qw,qx,qy,qz). Returns radians."""
    q = np.asarray(q, dtype=float)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def wrap_deg(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return (a + 180.0) % 360.0 - 180.0


def compute_metrics(
    rollout_obs: np.ndarray,
    gt_obs: np.ndarray,
    actions_pred: np.ndarray,
    actions_expert: np.ndarray,
    pos_diverge_thresh_m: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all Task 2 metrics for one trajectory rollout.

    Expected obs layout (unscaled):
      pos: 0:3
      vel: 3:6
      quat: 6:10 (qw,qx,qy,qz)
      setpoint: 10:13
      vbat: 13
    """
    roll = np.asarray(rollout_obs, dtype=float)
    gt = np.asarray(gt_obs, dtype=float)
    u = np.asarray(actions_pred, dtype=float)
    u_exp = np.asarray(actions_expert, dtype=float)

    valid = np.all(np.isfinite(roll[:, 0:6]), axis=1) & np.all(np.isfinite(gt[:, 0:6]), axis=1)
    n_valid = int(np.sum(valid))
    T = roll.shape[0]
    frac_valid = float(n_valid / T) if T > 0 else 0.0
    if n_valid < 2:
        return {
            "pos_RMSE_m": np.nan,
            "pos_ADE_m": np.nan,
            "pos_FDE_m": np.nan,
            "vel_RMSE_mps": np.nan,
            "heading_error_deg": np.nan,
            "divergence_rate_pct": 100.0,
            "action_MSE": float(np.nanmean((u - u_exp) ** 2)),
            "mean_valid_steps": float(n_valid),
            "frac_valid_steps": frac_valid,
        }

    pos_r = roll[valid, 0:3]
    pos_g = gt[valid, 0:3]
    vel_r = roll[valid, 3:6]
    vel_g = gt[valid, 3:6]

    e_pos = pos_errors(pos_r, pos_g)
    pos_rmse = rmse(e_pos)
    pos_ade = float(np.nanmean(e_pos))
    pos_fde = float(e_pos[-1])

    vel_err = np.linalg.norm(vel_r - vel_g, axis=1)
    vel_rmse = rmse(vel_err)

    yaw_r = np.degrees(yaw_from_quat_wxyz(roll[valid, 6:10]))
    yaw_g = np.degrees(yaw_from_quat_wxyz(gt[valid, 6:10]))
    yaw_err = wrap_deg(yaw_r - yaw_g)
    heading_error = float(np.nanmean(np.abs(yaw_err)))

    divergence_rate = float(np.nanmean(e_pos > pos_diverge_thresh_m) * 100.0)

    # Open-loop action MSE on test set
    action_mse = float(np.nanmean((u - u_exp) ** 2))

    return {
        "pos_RMSE_m": pos_rmse,
        "pos_ADE_m": pos_ade,
        "pos_FDE_m": pos_fde,
        "vel_RMSE_mps": vel_rmse,
        "heading_error_deg": heading_error,
        "divergence_rate_pct": divergence_rate,
        "action_MSE": action_mse,
        "mean_valid_steps": float(n_valid),
        "frac_valid_steps": frac_valid,
    }

