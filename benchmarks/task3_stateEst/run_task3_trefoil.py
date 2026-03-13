# -*- coding: utf-8 -*-
"""
run_task3_trefoil.py — Task 3 State Estimation on Trefoil Trajectories
======================================================================

Evaluates onboard EKF state estimation on trefoil knot trajectories
at three speed regimes (slow, medium, fast). All runs (Mellinger and PID)
are pooled per speed; metrics are mean ± std across all runs. Overlay plot
shows Vicon ground truth and EKF estimate for one run per speed on the same 3D plot.

Outputs
-------
  results/task3_trefoil_ate_comparison.pdf
  results/task3_trefoil_error_timeseries.pdf   (vertically stacked, averaged over all runs)
  results/task3_trefoil_error_2x2_with_overlay.pdf  (2x2: three error panels + Vicon vs EKF XY)
  results/task3_trefoil_trajectory_overlay.pdf (one panel per speed: Vicon + EKF on same plot)
  results/task3_trefoil_velocity_error.pdf
  results/tables/task3_trefoil_table.tex
  results/tables/task3_trefoil_table.csv
"""

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D

_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_BENCH.parent))  # benchmarks/ — for utils/ and task3_estimation.py

from utils.data_loader import TrajectoryData
from utils.metrics import compute_ate, compute_rte, umeyama_align
from utils.plotting import (
    setup_plotting,
    save_fig,
    SINGLE_COL,
    DOUBLE_COL,
    DOUBLE_COL_TALL,
    COLORS,
)
from task3_estimation import _angle_diff

logger = logging.getLogger(__name__)

CONTROLLER_DIRS = {
    "Mellinger": _BENCH / "mellinger_trefoil",
    "PID":       _BENCH / "pid_trefoil",
}

SPEED_ORDER = ["slow", "medium", "fast"]
CONTROLLER_ORDER = ["Mellinger", "PID"]

CTRL_COLORS = {
    "Mellinger": "#0072B2",
    "PID":       "#D55E00",
}
SPEED_MARKERS = {"slow": "o", "medium": "s", "fast": "D"}
SPEED_LINESTYLES = {"slow": "-", "medium": "--", "fast": "-."}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_trajectory_from_csv(csv_path: Path) -> Optional[TrajectoryData]:
    """Load a flat CSV file (e.g. B9_trefoil_slow_rep1.csv) into a TrajectoryData."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"Failed to read {csv_path}: {e}")
        return None

    required = ["t", "px", "py", "pz", "pwr_pm_vbat", "motor_motor_m1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"{csv_path.name}: missing columns {missing}, skipping")
        return None

    if len(df) < 500:
        logger.warning(f"{csv_path.name}: only {len(df)} rows, skipping")
        return None

    # Forward-fill battery voltage
    df["pwr_pm_vbat"] = df["pwr_pm_vbat"].replace(0.0, float("nan")).ffill().bfill()

    flight_mask = df["motor_motor_m1"] > 1000
    n_flight = int(flight_mask.sum())
    if n_flight < 100:
        logger.warning(f"{csv_path.name}: only {n_flight} flight rows, skipping")
        return None

    vbat = df["pwr_pm_vbat"]
    traj_id = csv_path.stem  # e.g. "B9_trefoil_slow_rep1"

    return TrajectoryData(
        path=str(csv_path),
        traj_id=traj_id,
        trajectory_type="trefoil",
        category="tracking",
        repetition=1,
        df=df,
        metadata={},
        mean_voltage=float(vbat.mean()),
        min_voltage=float(vbat.min()),
        max_voltage=float(vbat.max()),
        n_samples=len(df),
        n_flight_samples=n_flight,
    )


def load_trefoil_trajectories() -> Dict[str, List[TrajectoryData]]:
    """Load all trefoil trajectories from flat CSV files, keyed by controller name."""
    result = {}
    for ctrl_name, ctrl_dir in CONTROLLER_DIRS.items():
        trajs = []
        if not ctrl_dir.exists():
            logger.warning(f"Directory not found: {ctrl_dir}")
            continue
        for csv_file in sorted(ctrl_dir.glob("*.csv")):
            traj = _load_trajectory_from_csv(csv_file)
            if traj is not None:
                trajs.append(traj)
                logger.info(f"  Loaded {ctrl_name}: {traj.traj_id}")
            else:
                logger.warning(f"  Skipped {ctrl_name}: {csv_file.name}")
        result[ctrl_name] = trajs
        logger.info(f"  {ctrl_name}: {len(trajs)} trajectories loaded")
    return result


def _parse_speed(traj_id: str) -> str:
    """Extract speed label from trajectory ID like 'B9_trefoil_fast_rep01'."""
    tid = traj_id.lower()
    for s in SPEED_ORDER:
        if s in tid:
            return s
    return "unknown"


# ---------------------------------------------------------------------------
# Per-trajectory evaluation (streamlined for trefoil analysis)
# ---------------------------------------------------------------------------

def evaluate_single(traj: TrajectoryData) -> Optional[Dict]:
    """Run EKF evaluation on a single trajectory."""
    df = traj.flight_df
    if len(df) < 200:
        logger.warning(f"{traj.traj_id}: too few flight samples ({len(df)}), skipping")
        return None

    p_gt = df[["px", "py", "pz"]].values
    v_gt = df[["vx", "vy", "vz"]].values
    t = df["t"].values

    gt_roll = df["roll"].values
    gt_pitch = df["pitch"].values
    gt_yaw = df["yaw"].values

    p_ekf = df[["est_stateEstimate_x", "est_stateEstimate_y",
                "est_stateEstimate_z"]].values
    v_ekf = df[["est_stateEstimate_vx", "est_stateEstimate_vy",
                "est_stateEstimate_vz"]].values

    ekf_roll = df["att_stateEstimate_roll"].values
    ekf_pitch = df["att_stateEstimate_pitch"].values
    ekf_yaw = df["att_stateEstimate_yaw"].values

    vbat = df["pwr_pm_vbat"].values

    ate_ekf = compute_ate(p_ekf, p_gt, align=True)
    rte_ekf = compute_rte(p_ekf, p_gt, window_lengths_m=[1.0])

    v_err = np.linalg.norm(v_ekf - v_gt, axis=1)
    vel_rmse = float(np.sqrt(np.mean(v_err ** 2)))

    att_err_ekf = np.sqrt(
        (np.mean(_angle_diff(ekf_roll, gt_roll) ** 2) +
         np.mean(_angle_diff(ekf_pitch, gt_pitch) ** 2) +
         np.mean(_angle_diff(ekf_yaw, gt_yaw) ** 2)) / 3.0
    )

    _, _, p_ekf_aligned = umeyama_align(p_ekf, p_gt)
    e_xyz = p_ekf_aligned - p_gt
    pos_err_norm = np.linalg.norm(e_xyz, axis=1)

    t_rel = t - t[0]

    return {
        "traj_id":       traj.traj_id,
        "speed":         _parse_speed(traj.traj_id),
        "ate_ekf":       ate_ekf,
        "rte_ekf_1m":    rte_ekf.get("1.0", {}),
        "vel_rmse_ekf":  vel_rmse,
        "att_rmse_ekf":  float(att_err_ekf),
        "t_rel":         t_rel,
        "p_gt":          p_gt,
        "p_ekf":         p_ekf,
        "p_ekf_aligned": p_ekf_aligned,
        "e_xyz":         e_xyz,
        "pos_err":       pos_err_norm,
        "v_err":         v_err,
        "v_gt":          v_gt,
        "v_ekf":         v_ekf,
        "vbat":          vbat,
    }


# ---------------------------------------------------------------------------
# Aggregate and group
# ---------------------------------------------------------------------------

def group_results(
    all_results: Dict[str, List[Dict]],
) -> Dict[str, Dict[str, List[Dict]]]:
    """Organize results as  grouped[controller][speed] = [result_dicts]."""
    grouped = {}
    for ctrl, results in all_results.items():
        grouped[ctrl] = defaultdict(list)
        for r in results:
            grouped[ctrl][r["speed"]].append(r)
    return grouped


def group_by_speed(
    grouped: Dict[str, Dict[str, List[Dict]]],
) -> Dict[str, List[Dict]]:
    """Pool all runs by speed (ignore controller). by_speed[speed] = list of all result dicts."""
    by_speed = defaultdict(list)
    for ctrl in CONTROLLER_ORDER:
        for speed in SPEED_ORDER:
            by_speed[speed].extend(grouped.get(ctrl, {}).get(speed, []))
    return dict(by_speed)


def _interp_to_common_time(results: List[Dict], n_pts: int = 500):
    """Interpolate error signals onto a common normalized timeline [0, 1]."""
    t_norm = np.linspace(0.0, 1.0, n_pts)
    pos_errs = []
    vel_errs = []
    for r in results:
        t = r["t_rel"]
        if t[-1] - t[0] < 0.1:
            continue
        t_n = (t - t[0]) / (t[-1] - t[0])
        pos_errs.append(np.interp(t_norm, t_n, r["pos_err"]))
        vel_errs.append(np.interp(t_norm, t_n, r["v_err"]))
    return t_norm, np.array(pos_errs), np.array(vel_errs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ate_comparison(by_speed: Dict[str, List[Dict]]) -> plt.Figure:
    """Bar chart: EKF ATE (mean ± std) per speed, averaged over all runs."""
    fig, ax = plt.subplots(figsize=DOUBLE_COL)

    x_pos = np.arange(len(SPEED_ORDER))
    means, stds = [], []
    for speed in SPEED_ORDER:
        reps = by_speed.get(speed, [])
        ates = [r["ate_ekf"]["rmse"] * 1000 for r in reps]  # mm
        if ates:
            means.append(np.mean(ates))
            stds.append(np.std(ates))
        else:
            means.append(0)
            stds.append(0)

    ax.bar(x_pos, means, yerr=stds, capsize=4, color=COLORS["ekf"], alpha=0.85,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.capitalize() for s in SPEED_ORDER])
    ax.set_xlabel("Speed regime")
    ax.set_ylabel("EKF ATE RMSE (mm)")
    ax.set_title("EKF Position Accuracy on Trefoil (all runs pooled)")
    fig.tight_layout()
    return fig


def _y_fmt_k(x, pos):
    """Format y-axis: 10000 -> 10k, etc."""
    if abs(x) >= 1000:
        return f"{x / 1000:.0f}k"
    return f"{x:.0f}"


# Font sizes for compact single-col figures (match task2_control_il style)
_LABEL_SIZE = 8
_TICK_SIZE = 7
_TITLE_SIZE = 8


def plot_error_timeseries(by_speed: Dict[str, List[Dict]]) -> plt.Figure:
    """Position error over normalized flight time with std bands, per speed (vertically stacked, IEEE single col)."""
    w, h = SINGLE_COL[0], SINGLE_COL[1]  # 3.5 x 2.8 in
    fig, axes = plt.subplots(3, 1, figsize=(w, h), sharex=True, sharey=False)

    for ax, speed in zip(axes, SPEED_ORDER):
        reps = by_speed.get(speed, [])
        if not reps:
            continue
        t_norm, pos_errs, _ = _interp_to_common_time(reps)
        if len(pos_errs) == 0:
            continue
        mean_err = np.mean(pos_errs, axis=0) * 1000  # mm
        std_err = np.std(pos_errs, axis=0) * 1000

        ax.plot(t_norm * 100, mean_err, color=COLORS["ekf"], linewidth=1.2)
        ax.fill_between(t_norm * 100, mean_err - std_err, mean_err + std_err,
                        color=COLORS["ekf"], alpha=0.2)
        ax.set_title(speed.capitalize(), fontsize=_TITLE_SIZE)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    for ax in axes:
        ax.tick_params(axis="both", labelsize=_TICK_SIZE)
    axes[1].set_ylabel("Position error (mm)", fontsize=_LABEL_SIZE)
    axes[1].set_ylim(0, 50)
    axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(_y_fmt_k))
    axes[-1].set_xlabel("Flight progress (%)", fontsize=_LABEL_SIZE)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    return fig


def plot_error_2x2_with_overlay(by_speed: Dict[str, List[Dict]]) -> plt.Figure:
    """2x2: three position-error panels (slow, medium, fast) + 4th panel Vicon vs EKF (XY) trajectory. IEEE single-col style."""
    w = SINGLE_COL[0]
    fig, axes = plt.subplots(2, 2, figsize=(w, w), sharex=False)

    # --- Top-left (0,0): slow ---
    ax = axes[0, 0]
    reps = by_speed.get("slow", [])
    if reps:
        t_norm, pos_errs, _ = _interp_to_common_time(reps)
        if len(pos_errs) > 0:
            mean_err = np.mean(pos_errs, axis=0) * 1000
            std_err = np.std(pos_errs, axis=0) * 1000
            ax.plot(t_norm * 100, mean_err, color=COLORS["ekf"], linewidth=1.2)
            ax.fill_between(t_norm * 100, mean_err - std_err, mean_err + std_err,
                            color=COLORS["ekf"], alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # --- Top-right (0,1): medium ---
    ax = axes[0, 1]
    reps = by_speed.get("medium", [])
    if reps:
        t_norm, pos_errs, _ = _interp_to_common_time(reps)
        if len(pos_errs) > 0:
            mean_err = np.mean(pos_errs, axis=0) * 1000
            std_err = np.std(pos_errs, axis=0) * 1000
            ax.plot(t_norm * 100, mean_err, color=COLORS["ekf"], linewidth=1.2)
            ax.fill_between(t_norm * 100, mean_err - std_err, mean_err + std_err,
                            color=COLORS["ekf"], alpha=0.2)
    ax.set_ylim(0, 50)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_ylabel("Position error (mm)")

    # --- Bottom-left (1,0): fast ---
    ax = axes[1, 0]
    reps = by_speed.get("fast", [])
    if reps:
        t_norm, pos_errs, _ = _interp_to_common_time(reps)
        if len(pos_errs) > 0:
            mean_err = np.mean(pos_errs, axis=0) * 1000
            std_err = np.std(pos_errs, axis=0) * 1000
            ax.plot(t_norm * 100, mean_err, color=COLORS["ekf"], linewidth=1.2)
            ax.fill_between(t_norm * 100, mean_err - std_err, mean_err + std_err,
                            color=COLORS["ekf"], alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_y_fmt_k))
    ax.set_xlabel("Flight progress (%)")

    # --- Bottom-right (1,1): Vicon vs EKF (XY trajectory) ---
    ax = axes[1, 1]
    reps = by_speed.get("slow", [])
    if reps:
        rep = reps[0]
        p_gt = rep["p_gt"]
        p_ekf = rep["p_ekf"]
        ax.plot(p_gt[:, 0], p_gt[:, 1], color=COLORS["vicon"], linewidth=1.2, label="Vicon")
        ax.plot(p_ekf[:, 0], p_ekf[:, 1], color=COLORS["ekf"], linewidth=1.0, alpha=0.8, label="EKF")
        ax.scatter(p_gt[0, 0], p_gt[0, 1], c="green", s=20, zorder=5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 1])
    return fig


def plot_velocity_error_comparison(by_speed: Dict[str, List[Dict]]) -> plt.Figure:
    """Velocity error over normalized time with std bands, per speed (all runs pooled)."""
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL[0], 3.5), sharey=False)

    for ax, speed in zip(axes, SPEED_ORDER):
        reps = by_speed.get(speed, [])
        if not reps:
            continue
        t_norm, _, vel_errs = _interp_to_common_time(reps)
        if len(vel_errs) == 0:
            continue
        mean_err = np.mean(vel_errs, axis=0)
        std_err = np.std(vel_errs, axis=0)
        ax.plot(t_norm * 100, mean_err, color=COLORS["ekf"], linewidth=1.5)
        ax.fill_between(t_norm * 100, mean_err - std_err, mean_err + std_err,
                        color=COLORS["ekf"], alpha=0.2)
        ax.set_xlabel("Flight progress (%)")
        ax.set_title(f"{speed.capitalize()}")

    axes[0].set_ylabel("Velocity error (m/s)")
    fig.suptitle("EKF Velocity Error (mean ± std, all runs)", y=1.02)
    fig.tight_layout()
    return fig


def plot_trajectory_overlay_grid(by_speed: Dict[str, List[Dict]]) -> plt.Figure:
    """One 3D panel per speed: Vicon ground truth and EKF estimate for one run on the same plot."""
    fig = plt.figure(figsize=(DOUBLE_COL[0], 6.0))
    axes = []
    for i in range(len(SPEED_ORDER)):
        axes.append(fig.add_subplot(1, 3, i + 1, projection="3d"))

    for ax, speed in zip(axes, SPEED_ORDER):
        reps = by_speed.get(speed, [])
        if not reps:
            ax.text2D(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax.transAxes)
            continue
        rep = reps[0]
        p_gt = rep["p_gt"]
        p_ekf = rep["p_ekf"]
        ax.plot(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2],
                color=COLORS["vicon"], linewidth=1.5, label="Vicon (ground truth)")
        ax.plot(p_ekf[:, 0], p_ekf[:, 1], p_ekf[:, 2],
                color=COLORS["ekf"], linewidth=1.0, alpha=0.8, label="EKF (estimate)")
        ax.scatter(p_gt[0, 0], p_gt[0, 1], p_gt[0, 2], c="green", s=30, zorder=5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"{speed.capitalize()}")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Trefoil: Vicon vs EKF (one run per speed)", y=1.02)
    fig.tight_layout()
    return fig


def plot_per_axis_error(by_speed: Dict[str, List[Dict]]) -> plt.Figure:
    """Per-axis (x,y,z) error with std bands per speed (all runs pooled)."""
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL[0], 6.5), sharex=True)

    axis_labels = ["X error (mm)", "Y error (mm)", "Z error (mm)"]
    n_pts = 500
    t_norm = np.linspace(0.0, 1.0, n_pts)

    for col, speed in enumerate(SPEED_ORDER):
        reps = by_speed.get(speed, [])
        axis_errs = [[] for _ in range(3)]
        for r in reps:
            t = r["t_rel"]
            if t[-1] - t[0] < 0.1:
                continue
            t_n = (t - t[0]) / (t[-1] - t[0])
            for a in range(3):
                axis_errs[a].append(
                    np.interp(t_norm, t_n, r["e_xyz"][:, a]) * 1000
                )

        for row in range(3):
            ax = axes[row, col]
            if not axis_errs[row]:
                continue
            arr = np.array(axis_errs[row])
            mean_e = np.mean(arr, axis=0)
            std_e = np.std(arr, axis=0)
            ax.plot(t_norm * 100, mean_e, color=COLORS["ekf"], linewidth=1.2)
            ax.fill_between(t_norm * 100, mean_e - std_e, mean_e + std_e,
                            color=COLORS["ekf"], alpha=0.15)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
            if col == 0:
                ax.set_ylabel(axis_labels[row], fontsize=9)
        axes[0, col].set_title(f"{speed.capitalize()}")
        axes[2, col].set_xlabel("Flight progress (%)")
    fig.suptitle("Per-Axis EKF Error (mean ± std, all runs)", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def build_trefoil_table(by_speed: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Build summary table: one row per speed (all runs pooled)."""
    def _fmt(vals, scale=1.0, precision=3):
        arr = np.array(vals) * scale
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return "---"
        m, s = np.mean(finite), np.std(finite)
        return f"{m:.{precision}f} $\\pm$ {s:.{precision}f}"

    rows = []
    for speed in SPEED_ORDER:
        reps = by_speed.get(speed, [])
        if not reps:
            continue
        ate_ekf_vals = [r["ate_ekf"]["rmse"] for r in reps]
        ate_ekf_mean_vals = [r["ate_ekf"]["mean"] for r in reps]
        rte_ekf_vals = [r["rte_ekf_1m"].get("trans_mean", np.nan) for r in reps]
        vel_rmse_vals = [r["vel_rmse_ekf"] for r in reps]
        att_rmse_vals = [r["att_rmse_ekf"] for r in reps]
        rows.append({
            "Speed":            speed.capitalize(),
            "N":                len(reps),
            "ATE RMSE (mm)":   _fmt(ate_ekf_vals, scale=1000, precision=1),
            "ATE Mean (mm)":   _fmt(ate_ekf_mean_vals, scale=1000, precision=1),
            "RTE 1m (mm)":     _fmt(rte_ekf_vals, scale=1000, precision=1),
            "Vel RMSE (m/s)":  _fmt(vel_rmse_vals, precision=3),
            "Att RMSE (deg)":  _fmt(att_rmse_vals, precision=2),
        })
    return pd.DataFrame(rows)


def save_trefoil_table(df: pd.DataFrame, results_dir: Path) -> None:
    """Save table as CSV and LaTeX."""
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_dir / "task3_trefoil_table.csv", index=False)

    n_cols = len(df.columns)
    col_fmt = "@{}l" + "c" * (n_cols - 1) + "@{}"

    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Task~3: EKF state estimation accuracy on trefoil trajectories. "
                       r"Mean $\pm$ std across all runs (pooled).}")
    latex_lines.append(r"\label{tab:task3_trefoil}")
    latex_lines.append(r"\resizebox{\columnwidth}{!}{%")
    latex_lines.append(r"\begin{tabular}{" + col_fmt + "}")
    latex_lines.append(r"\toprule")

    header = " & ".join([r"\textbf{" + c + "}" for c in df.columns]) + r" \\"
    latex_lines.append(header)
    latex_lines.append(r"\midrule")

    for _, row in df.iterrows():
        vals = " & ".join(str(v) for v in row.values)
        latex_lines.append(vals + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table}")

    tex_path = tables_dir / "task3_trefoil_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")

    logger.info(f"  Saved {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    """Run full Task 3 trefoil evaluation."""
    setup_plotting()
    results_dir = Path(__file__).resolve().parent / "results"

    logger.info("=" * 60)
    logger.info("Task 3 Trefoil Evaluation")
    logger.info("=" * 60)

    traj_by_ctrl = load_trefoil_trajectories()

    all_results: Dict[str, List[Dict]] = {}
    total = 0
    for ctrl, trajs in traj_by_ctrl.items():
        ctrl_results = []
        for traj in trajs:
            logger.info(f"  Evaluating {ctrl}/{traj.traj_id} ...")
            res = evaluate_single(traj)
            if res is not None:
                ctrl_results.append(res)
        all_results[ctrl] = ctrl_results
        total += len(ctrl_results)
        logger.info(f"  {ctrl}: {len(ctrl_results)}/{len(trajs)} evaluated")

    if total == 0:
        logger.error("No valid results. Aborting.")
        return

    grouped = group_results(all_results)
    by_speed = group_by_speed(grouped)

    logger.info("Generating plots ...")

    fig1 = plot_ate_comparison(by_speed)
    p1 = save_fig(fig1, "task3_trefoil_ate_comparison.pdf")
    logger.info(f"  Saved {p1}")

    fig2 = plot_error_timeseries(by_speed)
    p2 = save_fig(fig2, "task3_trefoil_error_timeseries.pdf")
    logger.info(f"  Saved {p2}")

    fig2b = plot_error_2x2_with_overlay(by_speed)
    p2b = save_fig(fig2b, "task3_trefoil_error_2x2_with_overlay.pdf")
    logger.info(f"  Saved {p2b}")

    fig3 = plot_velocity_error_comparison(by_speed)
    p3 = save_fig(fig3, "task3_trefoil_velocity_error.pdf")
    logger.info(f"  Saved {p3}")

    fig4 = plot_trajectory_overlay_grid(by_speed)
    p4 = save_fig(fig4, "task3_trefoil_trajectory_overlay.pdf")
    logger.info(f"  Saved {p4}")

    fig5 = plot_per_axis_error(by_speed)
    p5 = save_fig(fig5, "task3_trefoil_per_axis_error.pdf")
    logger.info(f"  Saved {p5}")

    logger.info("Building table ...")
    table_df = build_trefoil_table(by_speed)
    save_trefoil_table(table_df, results_dir)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE (all runs pooled by speed)")
    print("=" * 70)
    print(table_df.to_string(index=False))
    print("=" * 70)

    logger.info("Task 3 trefoil evaluation complete.")
    return {"grouped": grouped, "by_speed": by_speed, "table": table_df}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run()
