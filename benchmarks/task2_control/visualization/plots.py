"""Task 2 plotting (figures + tables) with NanoBench IEEE style.

We intentionally reuse benchmarks/utils/plotting.py to match Task 1 style.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..evaluation.metrics import BATTERY_BINS

# Reuse existing benchmark-wide plotting conventions
from benchmarks.utils.plotting import (  # type: ignore
    setup_plotting,
    SINGLE_COL,
    DOUBLE_COL,
    DOUBLE_COL_TALL,
    COLORS,
)


def _out_dirs(task_root: Path) -> Tuple[Path, Path]:
    fig_dir = task_root / "results" / "figures"
    tab_dir = task_root / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, tab_dir


def save_pdf_png(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_training_curves(
    histories: Dict[str, Dict[str, List[float]]],
    early_stop_epoch: Optional[Dict[str, int]] = None,
    figsize=SINGLE_COL,
) -> plt.Figure:
    setup_plotting()
    fig, ax = plt.subplots(figsize=figsize)
    for name, h in histories.items():
        tr = h.get("train_loss", [])
        va = h.get("val_loss", [])
        if tr:
            ax.plot(np.arange(1, len(tr) + 1), tr, label=f"{name} train")
        if va:
            ax.plot(np.arange(1, len(va) + 1), va, linestyle="--", label=f"{name} val")
        if early_stop_epoch and name in early_stop_epoch:
            ax.axvline(early_stop_epoch[name], linestyle="--", color="k", alpha=0.4, linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)
    return fig


def plot_open_loop_motor_predictions(
    t_s: np.ndarray,
    motors_gt: np.ndarray,
    motors_pred: Dict[str, np.ndarray],
    figsize=DOUBLE_COL,
) -> plt.Figure:
    """4-subplot motor comparisons."""
    setup_plotting()
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axs = axs.flatten()
    labels = ["m1", "m2", "m3", "m4"]
    for i, ax in enumerate(axs):
        # Draw baselines first, then GT on top so it is visible
        for name, u in motors_pred.items():
            ax.plot(t_s, u[:, i], label=name, zorder=1)
        ax.plot(t_s, motors_gt[:, i], "k--", label="GT", zorder=3)
        ax.set_ylabel(labels[i])
        if i >= 2:
            ax.set_xlabel("Time [s]")
    handles, labs = axs[0].get_legend_handles_labels()
    # Ensure GT appears first in the legend
    order = sorted(range(len(labs)), key=lambda idx: 0 if labs[idx] == "GT" else 1)
    handles = [handles[i] for i in order]
    labs = [labs[i] for i in order]
    fig.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=min(len(labs), 5), frameon=False)
    fig.tight_layout()
    return fig


def plot_closed_loop_xyz(
    t_s: np.ndarray,
    pos_gt: np.ndarray,
    pos_rollouts: Dict[str, np.ndarray],
    figsize=DOUBLE_COL,
) -> plt.Figure:
    setup_plotting()

    # Truncate all series to the time window where MPPI has data, so that
    # the x-axis stops when MPPI diverges / becomes NaN.
    t_plot = t_s
    pos_gt_plot = pos_gt
    pos_rollouts_plot: Dict[str, np.ndarray] = {k: v for k, v in pos_rollouts.items()}
    if "MPPI" in pos_rollouts_plot:
        mppi = np.asarray(pos_rollouts_plot["MPPI"], dtype=float)
        valid = np.any(np.isfinite(mppi), axis=1)
        if np.any(valid):
            last = int(np.where(valid)[0][-1])
            t_plot = t_s[: last + 1]
            pos_gt_plot = pos_gt[: last + 1]
            for name in list(pos_rollouts_plot.keys()):
                pos_rollouts_plot[name] = pos_rollouts_plot[name][: last + 1]

    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
    axes = ["x", "y", "z"]
    for i, ax in enumerate(axs):
        ax.plot(t_plot, pos_gt_plot[:, i], "k--", label="GT", linewidth=2.5)
        for name, p in pos_rollouts_plot.items():
            ax.plot(t_plot, p[:, i], label=name)
        ax.set_ylabel(f"{axes[i]} [m]")
    axs[-1].set_xlabel("Time [s]")
    handles, labs = axs[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=min(len(labs), 4), frameon=False)
    fig.tight_layout()
    return fig


def plot_rmse_vs_battery(
    rmse_by_model_and_bin: Dict[str, Dict[str, List[float]]],
    figsize=SINGLE_COL,
) -> plt.Figure:
    setup_plotting()
    fig, ax = plt.subplots(figsize=figsize)
    bins = [b.name for b in BATTERY_BINS]
    x = np.arange(len(bins))
    width = 0.25
    models = list(rmse_by_model_and_bin.keys())
    for i, m in enumerate(models):
        vals = []
        stds = []
        for b in bins:
            arr = np.asarray(rmse_by_model_and_bin[m].get(b, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            vals.append(float(np.mean(arr)) if arr.size else float("nan"))
            stds.append(float(np.std(arr)) if arr.size else 0.0)
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width=width, yerr=stds, capsize=2, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("pos RMSE [m]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=min(len(models), 3), frameon=False)
    return fig


def plot_rmse_vs_trajectory_type(
    rmse_by_model_and_type: Dict[str, Dict[str, List[float]]],
    figsize=SINGLE_COL,
) -> plt.Figure:
    setup_plotting()
    fig, ax = plt.subplots(figsize=figsize)
    types = sorted({t for d in rmse_by_model_and_type.values() for t in d.keys()})
    x = np.arange(len(types))
    width = 0.25
    models = list(rmse_by_model_and_type.keys())
    for i, m in enumerate(models):
        vals = []
        stds = []
        for tt in types:
            arr = np.asarray(rmse_by_model_and_type[m].get(tt, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            vals.append(float(np.mean(arr)) if arr.size else float("nan"))
            stds.append(float(np.std(arr)) if arr.size else 0.0)
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width=width, yerr=stds, capsize=2, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=20, ha="right")
    ax.set_ylabel("pos RMSE [m]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=min(len(models), 3), frameon=False)
    fig.tight_layout()
    return fig


def plot_3d_trajectory(
    pos_gt: np.ndarray,
    pos_rollouts: Dict[str, np.ndarray],
    figsize=SINGLE_COL,
) -> plt.Figure:
    setup_plotting()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], color="k", linewidth=2.0, label="GT")
    for name, p in pos_rollouts.items():
        ax.plot(p[:, 0], p[:, 1], p[:, 2], linewidth=1.2, alpha=0.6, label=name)

    # Equal aspect
    mins = np.min(pos_gt, axis=0)
    maxs = np.max(pos_gt, axis=0)
    mid = 0.5 * (mins + maxs)
    r = 0.55 * float(np.max(maxs - mins) + 1e-6)
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def save_rep_trajectory_plot_data(
    out_dir: Path,
    t_s: np.ndarray,
    motors_gt: np.ndarray,
    motors_pred: Dict[str, np.ndarray],
    pos_gt: np.ndarray,
    pos_rollouts: Dict[str, np.ndarray],
    trajectory_name: str = "rep",
) -> None:
    """
    Save time-series CSVs for the representative trajectory so you can replot
    open-loop motor and closed-loop XYZ without rerunning eval.

    Writes:
      - task2_rep_open_loop_motors.csv: t_s, m1_gt, m2_gt, m3_gt, m4_gt,
        m1_BC-MLP, m2_BC-MLP, ... m4_BC-LSTM (one column per model per motor).
      - task2_rep_closed_loop_xyz.csv: t_s, x_gt, y_gt, z_gt,
        x_BC-MLP, y_BC-MLP, z_BC-MLP, ... (one column per model per axis).
    """
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(t_s)

    # Open-loop motors: t_s has length len(motors_gt); align to n
    m_gt = np.asarray(motors_gt, dtype=float)
    if m_gt.shape[0] != n:
        m_gt = m_gt[:n]
    t_open = t_s[: m_gt.shape[0]]
    cols_open: Dict[str, np.ndarray] = {"t_s": t_open, "m1_gt": m_gt[:, 0], "m2_gt": m_gt[:, 1], "m3_gt": m_gt[:, 2], "m4_gt": m_gt[:, 3]}
    for name, u in motors_pred.items():
        uu = np.asarray(u, dtype=float)[: len(t_open)]
        for i in range(4):
            cols_open[f"m{i+1}_{name}"] = uu[:, i]
    pd.DataFrame(cols_open).to_csv(out_dir / "task2_rep_open_loop_motors.csv", index=False)

    # Closed-loop XYZ: one row per timestep, GT and each model's x,y,z
    pos_g = np.asarray(pos_gt, dtype=float)[:n]
    cols_xyz: Dict[str, np.ndarray] = {"t_s": t_s[:n], "x_gt": pos_g[:, 0], "y_gt": pos_g[:, 1], "z_gt": pos_g[:, 2]}
    for name, p in pos_rollouts.items():
        pp = np.asarray(p, dtype=float)[:n]
        cols_xyz[f"x_{name}"] = pp[:, 0]
        cols_xyz[f"y_{name}"] = pp[:, 1]
        cols_xyz[f"z_{name}"] = pp[:, 2]
    pd.DataFrame(cols_xyz).to_csv(out_dir / "task2_rep_closed_loop_xyz.csv", index=False)


def save_tables_latex_csv(
    out_dir: Path,
    table_name: str,
    rows: List[Dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{table_name}.csv"
    tex_path = out_dir / f"{table_name}.tex"

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # simple booktabs style
    tex = df.to_latex(index=False, escape=False, longtable=False, float_format="%.4f", caption=None, label=None)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

