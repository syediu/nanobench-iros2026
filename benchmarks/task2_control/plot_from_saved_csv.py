#!/usr/bin/env python3
"""
plot_from_saved_csv.py

Replot Task 2 open-loop motors and closed-loop XYZ from the CSVs written by
save_rep_trajectory_plot_data(), with control over the time window.

Example:

    python benchmarks/task2_control/plot_from_saved_csv.py \
        --plot-data-dir benchmarks/task2_control/results/plot_data \
        --t-start 0.0 --t-end 12.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Ensure repository root is on sys.path (same trick as run_task2.py)
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent.parent  # .../nanobench
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.task2_control.visualization.plots import (
    _out_dirs,
    plot_open_loop_motor_predictions,
    plot_closed_loop_xyz,
    save_pdf_png,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plot-data-dir",
        type=str,
        default=None,
        help="Directory containing task2_rep_open_loop_motors.csv and "
             "task2_rep_closed_loop_xyz.csv "
             "(default: <task_root>/results/plot_data)",
    )
    ap.add_argument(
        "--t-start",
        type=float,
        default=None,
        help="Optional start time [s] for plotting window.",
    )
    ap.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="Optional end time [s] for plotting window.",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="custom",
        help="Tag appended to output figure filenames.",
    )
    ap.add_argument(
        "--pid-csv",
        type=str,
        default=None,
        help="Optional path to PID aligned.csv to overlay as an additional trajectory.",
    )
    ap.add_argument(
        "--mellinger-csv",
        type=str,
        default=None,
        help="Optional path to Mellinger aligned.csv to overlay as an additional trajectory.",
    )
    return ap.parse_args()


def _apply_time_window(df: pd.DataFrame, t_start: float | None, t_end: float | None) -> pd.DataFrame:
    if t_start is not None:
        df = df[df["t_s"] >= t_start]
    if t_end is not None:
        df = df[df["t_s"] <= t_end]
    return df.reset_index(drop=True)


def load_open_loop(
    csv_path: Path,
    t_start: float | None,
    t_end: float | None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load t_s, GT motors, and per-model motor predictions from CSV."""
    df = pd.read_csv(csv_path)
    df = _apply_time_window(df, t_start, t_end)

    t_s = df["t_s"].to_numpy(float)
    motors_gt = df[["m1_gt", "m2_gt", "m3_gt", "m4_gt"]].to_numpy(float)

    motors_pred: Dict[str, np.ndarray] = {}
    for col in df.columns:
        if col == "t_s":
            continue
        if col.endswith("_gt"):
            continue
        # Columns are like m1_BC-MLP, m2_BC-LSTM, etc.
        if col.startswith("m") and "_" in col:
            motor_name, model_name = col.split("_", 1)
            motor_idx = int(motor_name[1:]) - 1  # m1 -> 0, m2 -> 1, ...
            arr = motors_pred.setdefault(model_name, np.zeros((len(df), 4), dtype=float))
            arr[:, motor_idx] = df[col].to_numpy(float)

    return t_s, motors_gt, motors_pred


def load_closed_loop_xyz(
    csv_path: Path,
    t_start: float | None,
    t_end: float | None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load t_s, GT xyz, and per-model rollout xyz from CSV."""
    df = pd.read_csv(csv_path)
    df = _apply_time_window(df, t_start, t_end)

    t_s = df["t_s"].to_numpy(float)
    pos_gt = df[["x_gt", "y_gt", "z_gt"]].to_numpy(float)

    pos_rollouts: Dict[str, np.ndarray] = {}
    for col in df.columns:
        if col == "t_s":
            continue
        if col.endswith("_gt"):
            continue
        # Columns are like x_BC-MLP, y_MPPI, z_BC-LSTM, etc.
        if "_" in col and col[0] in ("x", "y", "z"):
            axis = col[0]
            model_name = col[2:]  # strip "x_" / "y_" / "z_"
            axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
            arr = pos_rollouts.setdefault(model_name, np.zeros((len(df), 3), dtype=float))
            arr[:, axis_idx] = df[col].to_numpy(float)

    return t_s, pos_gt, pos_rollouts


def _load_pid_mell_overlays(
    t_ref: np.ndarray,
    pid_csv: Path | None,
    mell_csv: Path | None,
) -> Dict[str, np.ndarray]:
    """
    Load PID / Mellinger aligned.csv files and resample px,py,pz onto t_ref.

    The aligned logs use an absolute time column 't' (seconds). We convert
    this to a relative time axis starting at zero, then linearly interpolate
    onto the closed-loop time grid used for the BC/MPPI curves.
    """
    overlays: Dict[str, np.ndarray] = {}

    def _resample(path: Path) -> np.ndarray:
        df = pd.read_csv(path)
        if "t" not in df or not {"px", "py", "pz"}.issubset(df.columns):
            raise ValueError(f"{path}: expected columns ['t','px','py','pz']")
        t_abs = df["t"].to_numpy(float)
        t = t_abs - t_abs[0]
        pos = df[["px", "py", "pz"]].to_numpy(float)
        out = np.zeros((len(t_ref), 3), dtype=float)
        for i, col in enumerate([0, 1, 2]):
            out[:, i] = np.interp(t_ref, t, pos[:, col])
        return out

    if pid_csv is not None and pid_csv.exists():
        try:
            overlays["PID"] = _resample(pid_csv)
        except Exception as e:
            print(f"WARNING: failed to load PID overlay from {pid_csv}: {e}")

    if mell_csv is not None and mell_csv.exists():
        try:
            overlays["Mellinger"] = _resample(mell_csv)
        except Exception as e:
            print(f"WARNING: failed to load Mellinger overlay from {mell_csv}: {e}")

    return overlays


def main() -> int:
    args = _parse_args()

    task_root = Path(__file__).resolve().parent
    default_plot_data_dir = task_root / "results" / "plot_data"
    plot_data_dir = Path(args.plot_data_dir) if args.plot_data_dir is not None else default_plot_data_dir

    # Default PID/Mellinger paths (can be overridden via CLI).
    pid_csv = Path(args.pid_csv) if args.pid_csv is not None else task_root / "pid_mell_data" / "pid_aligned.csv"
    mell_csv = Path(args.mellinger_csv) if args.mellinger_csv is not None else task_root / "pid_mell_data" / "mell_aligned.csv"

    open_loop_csv = plot_data_dir / "task2_rep_open_loop_motors.csv"
    closed_loop_csv = plot_data_dir / "task2_rep_closed_loop_xyz.csv"
    if not open_loop_csv.exists() or not closed_loop_csv.exists():
        raise SystemExit(
            f"Expected CSVs not found in {plot_data_dir}.\n"
            f"Missing: {open_loop_csv if not open_loop_csv.exists() else ''} "
            f"{closed_loop_csv if not closed_loop_csv.exists() else ''}"
        )

    print(f"Loading open-loop motors from: {open_loop_csv}")
    t_open, motors_gt, motors_pred = load_open_loop(open_loop_csv, args.t_start, args.t_end)

    print(f"Loading closed-loop XYZ from: {closed_loop_csv}")
    t_xyz, pos_gt, pos_rollouts = load_closed_loop_xyz(closed_loop_csv, args.t_start, args.t_end)

    # Add PID / Mellinger overlays, resampled onto the same time grid.
    extra_rollouts = _load_pid_mell_overlays(t_xyz, pid_csv if pid_csv.exists() else None, mell_csv if mell_csv.exists() else None)
    if extra_rollouts:
        print(f"Adding overlays: {', '.join(extra_rollouts.keys())}")
        pos_rollouts.update(extra_rollouts)

    fig_dir, _ = _out_dirs(task_root)

    # Open-loop motors
    fig_open = plot_open_loop_motor_predictions(
        t_s=t_open,
        motors_gt=motors_gt,
        motors_pred=motors_pred,
    )
    save_pdf_png(fig_open, fig_dir / f"fig_task2_open_loop_motor_{args.tag}")

    # Closed-loop XYZ
    fig_xyz = plot_closed_loop_xyz(
        t_s=t_xyz,
        pos_gt=pos_gt,
        pos_rollouts=pos_rollouts,
    )
    save_pdf_png(fig_xyz, fig_dir / f"fig_task2_closed_loop_xyz_{args.tag}")

    print(f"Saved figures to: {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())