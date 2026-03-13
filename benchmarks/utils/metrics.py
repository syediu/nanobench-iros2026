# -*- coding: utf-8 -*-
"""
metrics.py — Shared evaluation metrics for all three benchmark tasks.

All functions accept numpy arrays.  No pandas dependencies here.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


# ---------------------------------------------------------------------------
# Basic regression metrics
# ---------------------------------------------------------------------------

def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Root mean squared error."""
    pred = np.asarray(pred, dtype=float).ravel()
    gt = np.asarray(gt, dtype=float).ravel()
    mask = np.isfinite(pred) & np.isfinite(gt)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2)))


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean absolute error."""
    pred = np.asarray(pred, dtype=float).ravel()
    gt = np.asarray(gt, dtype=float).ravel()
    mask = np.isfinite(pred) & np.isfinite(gt)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - gt[mask])))


def r2_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Coefficient of determination R²."""
    pred = np.asarray(pred, dtype=float).ravel()
    gt = np.asarray(gt, dtype=float).ravel()
    mask = np.isfinite(pred) & np.isfinite(gt)
    if mask.sum() < 2:
        return float("nan")
    ss_res = np.sum((gt[mask] - pred[mask]) ** 2)
    ss_tot = np.sum((gt[mask] - np.mean(gt[mask])) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Voltage binning
# ---------------------------------------------------------------------------

VOLTAGE_BINS = [
    (4.0, float("inf"), ">4.0V (fresh)"),
    (3.8, 4.0, "3.8-4.0V (mid)"),
    (0.0, 3.8, "<3.8V (depleted)"),
]


def voltage_bin_label(v: float) -> str:
    """Return human-readable voltage bin label for a scalar voltage."""
    for lo, hi, label in VOLTAGE_BINS:
        if lo <= v < hi:
            return label
    return ">4.0V (fresh)"  # fallback


def bin_by_voltage(values: np.ndarray, voltages: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split values into three voltage bins.

    Returns dict keyed by bin label, values are 1-D arrays of the metric
    in that bin (may be empty).
    """
    result: Dict[str, List] = {label: [] for _, _, label in VOLTAGE_BINS}
    voltages = np.asarray(voltages, dtype=float)
    values = np.asarray(values, dtype=float)
    for v, val in zip(voltages, values):
        result[voltage_bin_label(v)].append(val)
    return {k: np.array(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Umeyama rigid-body alignment (Horn's method)
# Used for ATE computation
# ---------------------------------------------------------------------------

def umeyama_align(
    p_est: np.ndarray, p_gt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Least-squares rigid body alignment (Umeyama / Horn's method).

    Finds R, t such that  R @ p_est[i] + t ≈ p_gt[i]  in a least-squares sense.

    Parameters
    ----------
    p_est : (N, 3) estimated trajectory
    p_gt  : (N, 3) ground-truth trajectory

    Returns
    -------
    R         : (3, 3) rotation matrix
    t         : (3,) translation vector
    p_aligned : (N, 3) = (R @ p_est.T).T + t
    """
    p_est = np.asarray(p_est, dtype=float)
    p_gt = np.asarray(p_gt, dtype=float)
    assert p_est.shape == p_gt.shape, "p_est and p_gt must have the same shape"
    assert p_est.ndim == 2 and p_est.shape[1] == 3

    mu_est = p_est.mean(axis=0)
    mu_gt = p_gt.mean(axis=0)

    p_est_c = p_est - mu_est
    p_gt_c = p_gt - mu_gt

    # Cross-covariance matrix
    H = p_gt_c.T @ p_est_c  # (3, 3)

    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, d])

    R = U @ D @ Vt
    t = mu_gt - R @ mu_est

    p_aligned = (R @ p_est.T).T + t
    return R, t, p_aligned


# ---------------------------------------------------------------------------
# Absolute Trajectory Error (ATE)
# ---------------------------------------------------------------------------

def compute_ate(
    p_est: np.ndarray, p_gt: np.ndarray, align: bool = True
) -> Dict[str, float]:
    """
    Compute Absolute Trajectory Error after optional Umeyama alignment.

    Parameters
    ----------
    p_est  : (N, 3)
    p_gt   : (N, 3)
    align  : if True, align p_est to p_gt before computing errors

    Returns
    -------
    dict with keys: 'mean', 'std', 'rmse', 'max'
    """
    p_est = np.asarray(p_est, dtype=float)
    p_gt = np.asarray(p_gt, dtype=float)

    if len(p_est) < 3:
        nan = float("nan")
        return {"mean": nan, "std": nan, "rmse": nan, "max": nan}

    if align:
        _, _, p_aligned = umeyama_align(p_est, p_gt)
    else:
        p_aligned = p_est

    errors = np.linalg.norm(p_aligned - p_gt, axis=1)
    return {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max": float(np.max(errors)),
    }


# ---------------------------------------------------------------------------
# Relative Trajectory Error (RTE) — path-length based windows
# ---------------------------------------------------------------------------

def _cumulative_path_length(positions: np.ndarray) -> np.ndarray:
    """Compute cumulative path length along a trajectory."""
    diffs = np.diff(positions, axis=0)
    increments = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(increments)])


def compute_rte(
    p_est: np.ndarray,
    p_gt: np.ndarray,
    window_lengths_m: List[float] = [0.5, 1.0, 2.0],
) -> Dict[str, Dict[str, float]]:
    """
    Relative Trajectory Error over path-length windows.

    For each window length, extracts all sub-sequences of that path length
    from the ground-truth trajectory and computes mean/std translational error.

    Returns
    -------
    dict keyed by window length (float), each containing:
      'trans_mean', 'trans_std', 'rot_mean_deg', 'rot_std_deg', 'n_windows'
    """
    from scipy.spatial.transform import Rotation

    p_est = np.asarray(p_est, dtype=float)
    p_gt = np.asarray(p_gt, dtype=float)

    path_len = _cumulative_path_length(p_gt)
    N = len(p_gt)
    results: Dict[str, Dict[str, float]] = {}

    for win_m in window_lengths_m:
        trans_errors = []
        # Stride every 10 samples for efficiency
        for i in range(0, N - 1, 10):
            target_len = path_len[i] + win_m
            if target_len > path_len[-1]:
                break
            # Find end index
            j_arr = np.where(path_len >= target_len)[0]
            if len(j_arr) == 0:
                continue
            j = j_arr[0]
            if j >= N:
                continue
            # Translational error
            delta_gt = p_gt[j] - p_gt[i]
            delta_est = p_est[j] - p_est[i]
            trans_err = np.linalg.norm(delta_est - delta_gt)
            trans_errors.append(trans_err)

        key = str(win_m)
        if trans_errors:
            trans_arr = np.array(trans_errors)
            results[key] = {
                "trans_mean": float(np.mean(trans_arr)),
                "trans_std": float(np.std(trans_arr)),
                "n_windows": len(trans_arr),
            }
        else:
            results[key] = {
                "trans_mean": float("nan"),
                "trans_std": float("nan"),
                "n_windows": 0,
            }

    return results


# ---------------------------------------------------------------------------
# Windowed metrics (for voltage-vs-error analysis)
# ---------------------------------------------------------------------------

def sliding_window_metric(
    values: np.ndarray,
    window_size: int,
    step: int = 1,
    metric_fn=np.mean,
) -> np.ndarray:
    """Apply metric_fn to sliding windows of `values`."""
    n = len(values)
    out = []
    for start in range(0, n - window_size + 1, step):
        out.append(metric_fn(values[start : start + window_size]))
    return np.array(out)


def compute_windowed_ate(
    p_est: np.ndarray,
    p_gt: np.ndarray,
    window_samples: int = 1000,
    step_samples: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ATE in sliding windows.

    Returns
    -------
    window_centers : (M,) array of center indices
    window_ates    : (M,) array of ATE values [meters]
    """
    n = len(p_est)
    centers = []
    ates = []
    for start in range(0, n - window_samples + 1, step_samples):
        end = start + window_samples
        pe = p_est[start:end]
        pg = p_gt[start:end]
        ate = compute_ate(pe, pg, align=True)
        centers.append((start + end) // 2)
        ates.append(ate["rmse"])
    return np.array(centers), np.array(ates)


# ---------------------------------------------------------------------------
# Segment-level statistics (for table generation with n>1 runs)
# ---------------------------------------------------------------------------

def segment_stats(values: np.ndarray) -> Dict[str, float]:
    """Return mean, std, min, max of an array. Safe for length-1 arrays."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        nan = float("nan")
        return {"mean": nan, "std": 0.0, "min": nan, "max": nan, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": len(values),
    }
