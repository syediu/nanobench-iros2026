#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_classical_vs_learned.py

Benchmark classical controllers (PID, Mellinger) on real-world trefoil
trajectories against learned baselines (BC-MLP, BC-LSTM, MPPI) evaluated
in simulation through a learned dynamics model.

The comparison is structured as:
  - PID / Mellinger: real-world reference-tracking error
        (Vicon position vs. commanded setpoint)
  - BC-MLP / BC-LSTM / MPPI: simulated imitation error
        (rollout position vs. expert trajectory)

Both groups use position RMSE and MAE as the primary metrics, but the
evaluation domains differ. Tables and figures clearly annotate this.

Usage
-----
    cd <nanobench_root>
    python benchmarks/task2_control/benchmark_classical_vs_learned.py

Outputs
-------
    benchmarks/task2_control/results/tables/
        table_task2_combined_comparison.csv / .tex
        table_task2_classical_per_run.csv
        table_task2_classical_speed_summary.csv / .tex

    benchmarks/task2_control/results/figures/
        fig_task2_main_comparison_bar.pdf / .png
        fig_task2_speed_analysis.pdf / .png
        fig_task2_tracking_error_boxplot.pdf / .png
        fig_task2_tracking_xyz_pid.pdf / .png
        fig_task2_tracking_xyz_mellinger.pdf / .png
        fig_task2_error_timeseries.pdf / .png
        fig_task2_3d_trajectory.pdf / .png
        fig_task2_per_axis_rmse.pdf / .png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.utils.plotting import (
    setup_plotting,
    SINGLE_COL,
    SINGLE_COL_TALL,
    DOUBLE_COL,
    DOUBLE_COL_TALL,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────
TASK_ROOT = _THIS.parent
RESULTS_DIR = TASK_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"

SPEEDS = ["slow", "medium", "fast"]

CTRL_COLORS = {
    "PID":       "#0072B2",
    "Mellinger": "#009E73",
    "BC-MLP":    "#E69F00",
    "BC-LSTM":   "#D55E00",
    "MPPI":      "#CC79A7",
}
CTRL_HATCHES = {
    "PID":       "",
    "Mellinger": "",
    "BC-MLP":    "//",
    "BC-LSTM":   "//",
    "MPPI":      "//",
}
CTRL_ORDER = ["PID", "Mellinger", "BC-MLP", "BC-LSTM", "MPPI"]

SPEED_COLORS = {"slow": "#56B4E9", "medium": "#E69F00", "fast": "#D55E00"}

COMPARISON_METRICS = [
    "pos_RMSE_m",
    "pos_MAE_m",
    "pos_FDE_m",
    "vel_RMSE_mps",
    "heading_error_deg",
    "divergence_rate_pct",
]
METRIC_LABELS = {
    "pos_RMSE_m":           "Pos. RMSE (m)",
    "pos_MAE_m":            "Pos. MAE (m)",
    "pos_max_m":            "Pos. Max (m)",
    "pos_FDE_m":            "Pos. FDE (m)",
    "vel_RMSE_mps":         "Vel. RMSE (m/s)",
    "heading_error_deg":    "Heading Err. (°)",
    "divergence_rate_pct":  "Divergence (%)",
    "x_RMSE_m":             "x RMSE (m)",
    "y_RMSE_m":             "y RMSE (m)",
    "z_RMSE_m":             "z RMSE (m)",
}


# ──────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────
def _file_hash(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_runs(controller_dir: Path) -> Dict[str, List[Path]]:
    """Return {speed: [aligned.csv paths]}, de-duplicated by file hash."""
    runs: Dict[str, List[Path]] = {}
    for speed in SPEEDS:
        dirs = sorted(controller_dir.glob(f"B9_trefoil_{speed}_rep*"))
        csvs = [d / "aligned.csv" for d in dirs if (d / "aligned.csv").exists()]
        seen_hashes: Dict[str, Path] = {}
        unique: List[Path] = []
        for c in csvs:
            h = _file_hash(c)
            if h in seen_hashes:
                print(f"    WARNING: {c.parent.name} is a duplicate of "
                      f"{seen_hashes[h].parent.name}, skipping")
            else:
                seen_hashes[h] = c
                unique.append(c)
        if unique:
            runs[speed] = unique
    return runs


def _detect_trajectory_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Return the sub-DataFrame where the trajectory setpoints are active.

    During hover / takeoff / landing the setpoints are all zero.  The
    trajectory phase is the contiguous block where the setpoint magnitude
    exceeds a small threshold.
    """
    sp = df[["sp_ctrltarget_x", "sp_ctrltarget_y", "sp_ctrltarget_z"]].values
    mag = np.linalg.norm(sp, axis=1)
    active = mag > 0.05
    if not active.any():
        raise ValueError("No active trajectory phase detected (all setpoints ≈ 0)")
    idx = np.where(active)[0]
    return df.iloc[idx[0] : idx[-1] + 1].reset_index(drop=True)


def load_trajectory_data(csv_path: Path) -> pd.DataFrame:
    """Load aligned.csv, return only the trajectory-tracking phase."""
    df = pd.read_csv(csv_path)
    return _detect_trajectory_phase(df)


def compute_tracking_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute position-tracking metrics (actual Vicon pos vs setpoint).

    Returns dict with keys matching COMPARISON_METRICS plus per-axis info.
    """
    pos = df[["px", "py", "pz"]].values
    sp  = df[["sp_ctrltarget_x", "sp_ctrltarget_y", "sp_ctrltarget_z"]].values

    err = pos - sp
    err_norm = np.linalg.norm(err, axis=1)

    pos_rmse = float(np.sqrt(np.mean(err_norm ** 2)))
    pos_mae  = float(np.mean(err_norm))
    pos_max  = float(np.max(err_norm))
    pos_fde  = float(err_norm[-1])

    x_rmse = float(np.sqrt(np.mean(err[:, 0] ** 2)))
    y_rmse = float(np.sqrt(np.mean(err[:, 1] ** 2)))
    z_rmse = float(np.sqrt(np.mean(err[:, 2] ** 2)))

    # Velocity RMSE: numerical derivative of setpoint as reference
    t = df["t"].values
    dt = np.diff(t)
    dt = np.where(dt == 0, 1e-6, dt)
    vel_ref = np.diff(sp, axis=0) / dt[:, None]
    vel_act = df[["vx", "vy", "vz"]].values[:-1]
    vel_err = np.linalg.norm(vel_act - vel_ref, axis=1)
    vel_rmse = float(np.sqrt(np.mean(vel_err ** 2)))

    # Heading error
    yaw_act = df["yaw"].values
    yaw_ref = df["sp_ctrltarget_yaw"].values
    yaw_err = np.degrees(yaw_act - yaw_ref)
    yaw_err = (yaw_err + 180.0) % 360.0 - 180.0
    heading_error_deg = float(np.mean(np.abs(yaw_err)))

    # Divergence rate (% of timesteps with error > 0.5 m)
    divergence_rate_pct = float(np.mean(err_norm > 0.5) * 100.0)

    duration_s = float(t[-1] - t[0])

    return {
        "pos_RMSE_m":          pos_rmse,
        "pos_MAE_m":           pos_mae,
        "pos_max_m":           pos_max,
        "pos_FDE_m":           pos_fde,
        "x_RMSE_m":            x_rmse,
        "y_RMSE_m":            y_rmse,
        "z_RMSE_m":            z_rmse,
        "vel_RMSE_mps":        vel_rmse,
        "heading_error_deg":   heading_error_deg,
        "divergence_rate_pct": divergence_rate_pct,
        "duration_s":          duration_s,
        "n_samples":           len(df),
    }


def load_baseline_trefoil(tab_dir: Path) -> pd.DataFrame:
    """Load per-trajectory baseline results and keep trefoil rows only."""
    csv = tab_dir / "task2_closed_loop_metrics_per_traj.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv)
    mask = df["trajectory"].str.contains("trefoil", case=False)
    return df[mask].copy()


def load_baseline_main(tab_dir: Path) -> pd.DataFrame:
    csv = tab_dir / "table_task2_closed_loop_main.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    return pd.read_csv(csv)


# ──────────────────────────────────────────────────────────────────────────
# Table generation
# ──────────────────────────────────────────────────────────────────────────
def _ms(mean: float, std: float, fmt: str = ".3f") -> str:
    return f"{mean:{fmt}} ± {std:{fmt}}"


def _bold_best(vals: List[float], lower_better: bool = True) -> List[bool]:
    """Return mask indicating which value is the best."""
    arr = np.array(vals, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return [False] * len(vals)
    best = np.nanargmin(arr) if lower_better else np.nanargmax(arr)
    out = [False] * len(vals)
    out[best] = True
    return out


def build_combined_table(
    classical: Dict[str, Dict[str, List[Dict[str, float]]]],
    baseline_trefoil: pd.DataFrame,
    speed: str = "medium",
) -> pd.DataFrame:
    """Build the main comparison table (classical at *speed* vs learned on trefoil)."""
    rows: List[Dict[str, Any]] = []

    for ctrl in ["PID", "Mellinger"]:
        runs = classical.get(ctrl, {}).get(speed, [])
        if not runs:
            continue
        row: Dict[str, Any] = {
            "Controller": ctrl,
            "Domain": "Real",
            "n_runs": len(runs),
        }
        for m in COMPARISON_METRICS:
            vals = [r[m] for r in runs if np.isfinite(r.get(m, np.nan))]
            row[m]            = float(np.mean(vals)) if vals else np.nan
            row[m + "_std"]   = float(np.std(vals))  if len(vals) > 1 else 0.0
        row["pos_max_m"]      = float(np.mean([r["pos_max_m"] for r in runs]))
        row["pos_max_m_std"]  = float(np.std([r["pos_max_m"] for r in runs])) if len(runs) > 1 else 0.0
        rows.append(row)

    for model in ["BC-MLP", "BC-LSTM", "MPPI"]:
        mdf = baseline_trefoil[baseline_trefoil["model"] == model]
        if mdf.empty:
            continue
        row = {
            "Controller": model,
            "Domain": "Sim",
            "n_runs": len(mdf),
        }
        for m in COMPARISON_METRICS:
            col = "pos_ADE_m" if m == "pos_MAE_m" else m
            if col in mdf.columns:
                row[m]          = float(mdf[col].mean())
                row[m + "_std"] = float(mdf[col].std()) if len(mdf) > 1 else 0.0
            else:
                row[m]          = np.nan
                row[m + "_std"] = 0.0
        row["pos_max_m"]     = np.nan
        row["pos_max_m_std"] = 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def build_speed_summary(
    classical: Dict[str, Dict[str, List[Dict[str, float]]]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ctrl in ["PID", "Mellinger"]:
        for speed in SPEEDS:
            runs = classical.get(ctrl, {}).get(speed, [])
            if not runs:
                continue
            row: Dict[str, Any] = {
                "Controller": ctrl,
                "Speed": speed,
                "n_runs": len(runs),
            }
            for m in ["pos_RMSE_m", "pos_MAE_m", "pos_max_m",
                       "x_RMSE_m", "y_RMSE_m", "z_RMSE_m",
                       "vel_RMSE_mps", "heading_error_deg"]:
                vals = [r[m] for r in runs if np.isfinite(r.get(m, np.nan))]
                row[m]          = float(np.mean(vals)) if vals else np.nan
                row[m + "_std"] = float(np.std(vals))  if len(vals) > 1 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def _latex_combined(df: pd.DataFrame) -> str:
    """Generate a publication-ready LaTeX table with booktabs."""
    cols = ["pos_RMSE_m", "pos_MAE_m", "pos_max_m", "pos_FDE_m",
            "vel_RMSE_mps", "heading_error_deg"]
    col_heads = [
        r"RMSE\,(m)\,$\downarrow$",
        r"MAE\,(m)\,$\downarrow$",
        r"Max\,(m)\,$\downarrow$",
        r"FDE\,(m)\,$\downarrow$",
        r"Vel.\,RMSE\,(m/s)\,$\downarrow$",
        r"Head.\,Err.\,($^\circ$)\,$\downarrow$",
    ]

    lines = [
        r"\begin{table}[t]",
        r"\caption{Tracking performance on the trefoil trajectory. "
        r"\emph{Real}: actual flight with Vicon; "
        r"\emph{Sim}: closed-loop rollout through a learned dynamics model.}",
        r"\label{tab:task2_combined}",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll" + "c" * len(cols) + "}",
        r"\toprule",
        "Controller & Domain & " + " & ".join(col_heads) + r" \\",
        r"\midrule",
    ]

    best = {}
    for c in cols:
        if c in df.columns:
            vals = df[c].tolist()
            bm = _bold_best(vals, lower_better=True)
            best[c] = bm
        else:
            best[c] = [False] * len(df)

    prev_domain = None
    for i, (_, r) in enumerate(df.iterrows()):
        if prev_domain is not None and r["Domain"] != prev_domain:
            lines.append(r"\midrule")
        prev_domain = r["Domain"]

        cells = [r["Controller"], r["Domain"]]
        for j, c in enumerate(cols):
            val = r.get(c, np.nan)
            std = r.get(c + "_std", 0.0)
            if np.isnan(val):
                cells.append("---")
            else:
                txt = f"{val:.3f}"
                if std > 0:
                    txt = f"{val:.3f}" + r"\,{\scriptsize$\pm$\," + f"{std:.3f}" + "}"
                if best[c][i]:
                    txt = r"\textbf{" + txt + "}"
                cells.append(txt)
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _latex_speed(df: pd.DataFrame) -> str:
    cols = ["pos_RMSE_m", "pos_MAE_m", "pos_max_m"]
    col_heads = [
        r"RMSE\,(m)\,$\downarrow$",
        r"MAE\,(m)\,$\downarrow$",
        r"Max\,(m)\,$\downarrow$",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Classical controller tracking error across trajectory speeds.}",
        r"\label{tab:task2_speed}",
        r"\centering",
        r"\begin{tabular}{llc" + "c" * len(cols) + "}",
        r"\toprule",
        "Controller & Speed & $n$ & " + " & ".join(col_heads) + r" \\",
        r"\midrule",
    ]
    prev_ctrl = None
    for _, r in df.iterrows():
        if prev_ctrl is not None and r["Controller"] != prev_ctrl:
            lines.append(r"\midrule")
        prev_ctrl = r["Controller"]
        cells = [r["Controller"], r["Speed"].capitalize(), str(int(r["n_runs"]))]
        for c in cols:
            val = r.get(c, np.nan)
            std = r.get(c + "_std", 0.0)
            if np.isnan(val):
                cells.append("---")
            elif std > 0:
                cells.append(f"{val:.4f}" + r"\,{\scriptsize$\pm$\," + f"{std:.4f}" + "}")
            else:
                cells.append(f"{val:.4f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: {path.with_suffix('.pdf')}")


def plot_main_comparison_bar(combined: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart: position RMSE for all controllers."""
    setup_plotting()
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    ctrls = combined["Controller"].tolist()
    rmses = combined["pos_RMSE_m"].tolist()
    stds  = [combined[f"pos_RMSE_m_std"].iloc[i] for i in range(len(ctrls))]

    x = np.arange(len(ctrls))
    colors  = [CTRL_COLORS.get(c, "#333") for c in ctrls]
    hatches = [CTRL_HATCHES.get(c, "")    for c in ctrls]

    bars = ax.bar(x, rmses, yerr=stds, capsize=3, color=colors,
                  edgecolor="black", linewidth=0.6, width=0.55, zorder=3)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(ctrls, fontsize=9)
    ax.set_ylabel("Position RMSE (m)")

    real_patch = mpatches.Patch(facecolor="white", edgecolor="black", label="Real-world")
    sim_patch  = mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="Simulated")
    ax.legend(handles=[real_patch, sim_patch], frameon=False, fontsize=8,
              loc="upper left")

    fig.tight_layout()
    return fig


def plot_multi_metric_bar(combined: pd.DataFrame) -> plt.Figure:
    """Side-by-side bars for RMSE, MAE, FDE across all controllers."""
    setup_plotting()
    metrics = ["pos_RMSE_m", "pos_MAE_m", "pos_FDE_m"]
    labels  = ["RMSE", "MAE", "FDE"]

    ctrls = combined["Controller"].tolist()
    n_m = len(metrics)
    n_c = len(ctrls)
    width = 0.8 / n_m

    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    x = np.arange(n_c)

    for j, (met, lab) in enumerate(zip(metrics, labels)):
        vals = combined[met].tolist()
        stds = combined.get(met + "_std", pd.Series([0.0] * n_c)).tolist()
        offset = (j - (n_m - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width, yerr=stds, capsize=2,
                      label=lab, zorder=3, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(ctrls, fontsize=10)
    ax.set_ylabel("Error (m)")
    ax.legend(frameon=False, ncol=n_m, loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_speed_analysis(
    classical: Dict[str, Dict[str, List[Dict[str, float]]]],
) -> plt.Figure:
    """Bar chart: PID vs Mellinger RMSE across speeds."""
    setup_plotting()
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    controllers = [c for c in ["PID", "Mellinger"] if c in classical]
    n_c = len(controllers)
    n_s = len(SPEEDS)
    width = 0.8 / n_c
    x = np.arange(n_s)

    for i, ctrl in enumerate(controllers):
        means, stds = [], []
        for speed in SPEEDS:
            runs = classical.get(ctrl, {}).get(speed, [])
            vals = [r["pos_RMSE_m"] for r in runs]
            means.append(float(np.mean(vals)) if vals else 0.0)
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        offset = (i - (n_c - 1) / 2) * width
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=3,
               color=CTRL_COLORS[ctrl], edgecolor="black", linewidth=0.5,
               label=ctrl, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SPEEDS])
    ax.set_xlabel("Trajectory Speed")
    ax.set_ylabel("Position RMSE (m)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_tracking_xyz(
    df: pd.DataFrame, controller_name: str, run_label: str,
) -> plt.Figure:
    """3-panel XYZ: actual vs reference for one run."""
    setup_plotting()
    fig, axs = plt.subplots(3, 1, figsize=DOUBLE_COL, sharex=True)

    t = df["t"].values - df["t"].values[0]
    pairs = [
        ("px", "sp_ctrltarget_x", "x"),
        ("py", "sp_ctrltarget_y", "y"),
        ("pz", "sp_ctrltarget_z", "z"),
    ]
    for ax, (act, ref, label) in zip(axs, pairs):
        ax.plot(t, df[ref].values, "k--", linewidth=1.0, label="Reference", zorder=2)
        ax.plot(t, df[act].values, color=CTRL_COLORS[controller_name],
                linewidth=1.2, label=controller_name, zorder=3)
        ax.set_ylabel(f"{label} (m)")

    axs[-1].set_xlabel("Time (s)")
    handles, labs = axs[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False)
    fig.suptitle(f"{controller_name} — {run_label}", y=1.05, fontsize=11)
    fig.tight_layout()
    return fig


def plot_error_timeseries(
    data: Dict[str, pd.DataFrame],
) -> plt.Figure:
    """Overlay position-error magnitude for PID and Mellinger representative runs."""
    setup_plotting()
    fig, ax = plt.subplots(figsize=DOUBLE_COL)

    for ctrl, df in data.items():
        t = df["t"].values - df["t"].values[0]
        err = np.linalg.norm(
            df[["px", "py", "pz"]].values -
            df[["sp_ctrltarget_x", "sp_ctrltarget_y", "sp_ctrltarget_z"]].values,
            axis=1,
        )
        ax.plot(t, err * 100, color=CTRL_COLORS.get(ctrl, "#333"),
                linewidth=1.2, label=ctrl, alpha=0.85)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (cm)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_3d_trajectory(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """3D view of actual vs reference for PID and Mellinger."""
    setup_plotting()
    fig = plt.figure(figsize=SINGLE_COL_TALL)
    ax = fig.add_subplot(111, projection="3d")

    first_df = list(data.values())[0]
    ref = first_df[["sp_ctrltarget_x", "sp_ctrltarget_y", "sp_ctrltarget_z"]].values
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], "k--", linewidth=1.5,
            label="Reference", alpha=0.6, zorder=1)

    for ctrl, df in data.items():
        pos = df[["px", "py", "pz"]].values
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                color=CTRL_COLORS.get(ctrl, "#333"),
                linewidth=1.2, label=ctrl, alpha=0.8, zorder=2)

    mins = np.min(ref, axis=0)
    maxs = np.max(ref, axis=0)
    mid = 0.5 * (mins + maxs)
    r = 0.55 * float(np.max(maxs - mins) + 1e-6)
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08),
              ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_boxplot(
    classical: Dict[str, Dict[str, List[Dict[str, float]]]],
    metric: str = "pos_RMSE_m",
) -> plt.Figure:
    """Box plot of tracking RMSE across runs, grouped by controller and speed."""
    setup_plotting()
    fig, ax = plt.subplots(figsize=DOUBLE_COL)

    groups: List[List[float]] = []
    labels: List[str] = []
    colors: List[str] = []

    for ctrl in ["PID", "Mellinger"]:
        for speed in SPEEDS:
            runs = classical.get(ctrl, {}).get(speed, [])
            vals = [r[metric] for r in runs if np.isfinite(r.get(metric, np.nan))]
            if not vals:
                continue
            groups.append(vals)
            labels.append(f"{ctrl}\n{speed.capitalize()}")
            colors.append(CTRL_COLORS[ctrl])

    if not groups:
        fig.text(0.5, 0.5, "No data", ha="center")
        return fig

    bp = ax.boxplot(groups, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=1.0),
                    capprops=dict(linewidth=1.0),
                    flierprops=dict(marker="o", markersize=4, alpha=0.6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
        patch.set_edgecolor("black")

    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    fig.tight_layout()
    return fig


def plot_per_axis_rmse(
    classical: Dict[str, Dict[str, List[Dict[str, float]]]],
    speed: str = "medium",
) -> plt.Figure:
    """Per-axis RMSE bar chart for PID and Mellinger at a given speed."""
    setup_plotting()
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    axes_metrics = ["x_RMSE_m", "y_RMSE_m", "z_RMSE_m"]
    axes_labels = ["x", "y", "z"]
    controllers = [c for c in ["PID", "Mellinger"] if speed in classical.get(c, {})]
    n_c = len(controllers)
    n_a = len(axes_metrics)
    width = 0.8 / n_c
    x = np.arange(n_a)

    for i, ctrl in enumerate(controllers):
        runs = classical[ctrl][speed]
        means, stds = [], []
        for m in axes_metrics:
            vals = [r[m] for r in runs]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        offset = (i - (n_c - 1) / 2) * width
        ax.bar(x + offset, [v * 100 for v in means], width=width,
               yerr=[v * 100 for v in stds], capsize=3,
               color=CTRL_COLORS[ctrl], edgecolor="black", linewidth=0.5,
               label=ctrl, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(axes_labels)
    ax.set_xlabel("Axis")
    ax.set_ylabel("RMSE (cm)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=str,
                    default=str(_REPO_ROOT / "datasets" / "nanobench_v1"))
    ap.add_argument("--comparison-speed", type=str, default="medium",
                    choices=SPEEDS,
                    help="Speed level used for the main combined comparison.")
    ap.add_argument("--outlier-threshold", type=float, default=1.0,
                    help="Runs with pos RMSE above this (m) are flagged as "
                         "failed flights and excluded. Set to 0 to disable.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Classical vs. Learned Controller Benchmarking")
    print("  Trajectory: Trefoil (B9)")
    print("=" * 64)

    # ── 1. Load classical controller data ─────────────────────────────
    controller_dirs = {
        "PID":       dataset_root / "pid_trefoil",
        "Mellinger": dataset_root / "mellinger_trefoil",
    }

    classical: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    representative: Dict[str, pd.DataFrame] = {}

    for ctrl, cdir in controller_dirs.items():
        if not cdir.exists():
            print(f"WARNING: {cdir} does not exist, skipping {ctrl}")
            continue
        print(f"\n── {ctrl} ──")
        runs_by_speed = discover_runs(cdir)
        classical[ctrl] = {}

        for speed, csv_paths in runs_by_speed.items():
            metrics_list: List[Dict[str, float]] = []
            print(f"  {speed}: {len(csv_paths)} unique run(s)")
            for cp in csv_paths:
                try:
                    df = load_trajectory_data(cp)
                    m = compute_tracking_metrics(df)
                    m["run"] = cp.parent.name
                    m["speed"] = speed
                    m["controller"] = ctrl
                    m["_csv_path"] = str(cp)

                    rmse = m["pos_RMSE_m"]
                    tag = ""
                    if args.outlier_threshold > 0 and rmse > args.outlier_threshold:
                        tag = "  ← EXCLUDED (likely failed flight)"
                    else:
                        metrics_list.append(m)
                        if speed == args.comparison_speed and ctrl not in representative:
                            representative[ctrl] = df

                    print(f"    {cp.parent.name}: "
                          f"RMSE={rmse:.4f} m  "
                          f"MAE={m['pos_MAE_m']:.4f} m  "
                          f"Max={m['pos_max_m']:.4f} m{tag}")
                except Exception as exc:
                    print(f"    ERROR {cp.parent.name}: {exc}")
            classical[ctrl][speed] = metrics_list

    # ── 2. Load learned-baseline results ──────────────────────────────
    print("\n── Learned baselines ──")
    try:
        baseline_trefoil = load_baseline_trefoil(TAB_DIR)
        baseline_main    = load_baseline_main(TAB_DIR)
        print(f"  Trefoil per-traj rows: {len(baseline_trefoil)}")
        print(f"  Main table rows:       {len(baseline_main)}")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        baseline_trefoil = pd.DataFrame()
        baseline_main    = pd.DataFrame()

    # ── 3. Build & save tables ────────────────────────────────────────
    print("\n── Tables ──")
    combined = build_combined_table(classical, baseline_trefoil,
                                    speed=args.comparison_speed)
    combined.to_csv(TAB_DIR / "table_task2_combined_comparison.csv", index=False)
    tex = _latex_combined(combined)
    (TAB_DIR / "table_task2_combined_comparison.tex").write_text(tex, encoding="utf-8")
    print(f"  Combined table ({len(combined)} rows) → "
          f"{TAB_DIR / 'table_task2_combined_comparison.csv'}")

    # Per-run classical
    all_runs = []
    for ctrl in ["PID", "Mellinger"]:
        for speed in SPEEDS:
            for r in classical.get(ctrl, {}).get(speed, []):
                all_runs.append(r)
    if all_runs:
        pd.DataFrame(all_runs).to_csv(
            TAB_DIR / "table_task2_classical_per_run.csv", index=False)

    speed_df = build_speed_summary(classical)
    if not speed_df.empty:
        speed_df.to_csv(TAB_DIR / "table_task2_classical_speed_summary.csv",
                        index=False)
        tex_s = _latex_speed(speed_df)
        (TAB_DIR / "table_task2_classical_speed_summary.tex").write_text(
            tex_s, encoding="utf-8")
        print(f"  Speed summary ({len(speed_df)} rows)")

    # ── 4. Print combined table to console ────────────────────────────
    print("\n" + "=" * 64)
    print("  Combined Comparison (trefoil, speed = {})".format(
        args.comparison_speed))
    print("=" * 64)
    display_cols = ["Controller", "Domain", "n_runs",
                    "pos_RMSE_m", "pos_MAE_m", "pos_max_m",
                    "pos_FDE_m", "vel_RMSE_mps", "heading_error_deg"]
    existing = [c for c in display_cols if c in combined.columns]
    print(combined[existing].to_string(index=False, float_format="%.4f"))

    # ── 5. Figures ────────────────────────────────────────────────────
    print("\n── Figures ──")

    fig = plot_main_comparison_bar(combined)
    _save(fig, FIG_DIR / "fig_task2_main_comparison_bar")

    fig = plot_multi_metric_bar(combined)
    _save(fig, FIG_DIR / "fig_task2_multi_metric_bar")

    if classical:
        fig = plot_speed_analysis(classical)
        _save(fig, FIG_DIR / "fig_task2_speed_analysis")

        fig = plot_boxplot(classical, metric="pos_RMSE_m")
        _save(fig, FIG_DIR / "fig_task2_tracking_error_boxplot")

        fig = plot_per_axis_rmse(classical, speed=args.comparison_speed)
        _save(fig, FIG_DIR / "fig_task2_per_axis_rmse")

    for ctrl, df in representative.items():
        run_label = f"trefoil {args.comparison_speed}"
        fig = plot_tracking_xyz(df, ctrl, run_label)
        _save(fig, FIG_DIR / f"fig_task2_tracking_xyz_{ctrl.lower()}")

    if representative:
        fig = plot_error_timeseries(representative)
        _save(fig, FIG_DIR / "fig_task2_error_timeseries")

        fig = plot_3d_trajectory(representative)
        _save(fig, FIG_DIR / "fig_task2_3d_trajectory")

    # ── 6. Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Interpretation Notes")
    print("=" * 64)
    print("""
  • PID and Mellinger are classical controllers evaluated on REAL hardware.
    Their error = |Vicon position − reference setpoint|.

  • BC-MLP, BC-LSTM, MPPI are learned controllers evaluated in SIMULATION
    via closed-loop rollout through a learned dynamics model.
    Their error = |simulated rollout position − expert trajectory|.

  • Both measure position-tracking fidelity, but in different domains.
    Classical controllers set an empirical performance ceiling; learned
    controllers attempt to replicate expert behaviour offline.

  • A direct numeric comparison is informative but not strictly
    apples-to-apples.  Deploying learned controllers on real hardware
    would enable a fair head-to-head evaluation.
""")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
