# -*- coding: utf-8 -*-
"""
plotting.py — Shared IEEE-ready plotting style for all NanoBench benchmark tasks.

Apply setup_plotting() once at the start of each task script.
All figures must be sized for IEEE double-column format.
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Output directory (created at runtime)
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ---------------------------------------------------------------------------
# IEEE figure sizes (inches)
# ---------------------------------------------------------------------------
SINGLE_COL = (3.5, 2.8)       # single-column figure
SINGLE_COL_TALL = (3.5, 3.5)  # single-column, square
DOUBLE_COL = (7.16, 3.2)      # double-column figure
DOUBLE_COL_TALL = (7.16, 4.8) # double-column, taller (3D plots, multi-panel)

# ---------------------------------------------------------------------------
# Colorblind-safe palette (Wong 2011)
# ---------------------------------------------------------------------------
COLORS = {
    "baseline_const":   "#E69F00",   # orange
    "baseline_time":    "#56B4E9",   # sky blue
    "baseline_linear":  "#009E73",   # green
    "our_model":        "#D55E00",   # vermillion (red-orange)
    "vicon":            "#000000",   # black
    "ekf":              "#0072B2",   # blue
    "dead_reckoning":   "#CC79A7",   # pink
    "complementary":    "#F0E442",   # yellow
    "setpoint":         "#999999",   # grey
    "voltage":          "#E69F00",   # orange (same as baseline_const)
    "error":            "#D55E00",   # vermillion
    "circle":           "#0072B2",
    "figure_eight":     "#D55E00",
    "hover":            "#009E73",
    "lemniscate":       "#CC79A7",
    "helix":            "#56B4E9",
    "default":          "#333333",
}

TRAJ_TYPE_COLORS = {
    "circle":       COLORS["circle"],
    "figure_eight": COLORS["figure_eight"],
    "hover":        COLORS["hover"],
    "lemniscate":   COLORS["lemniscate"],
    "helix":        COLORS["helix"],
}

MODEL_COLORS = {
    "Constant gain":        COLORS["baseline_const"],
    "Time-indexed":         COLORS["baseline_time"],
    "Linear regression":    COLORS["baseline_linear"],
    "Voltage-conditioned":  COLORS["our_model"],
}

MODEL_LINESTYLES = {
    "Constant gain":        "--",
    "Time-indexed":         "-.",
    "Linear regression":    ":",
    "Voltage-conditioned":  "-",
}


def setup_plotting() -> None:
    """Apply IEEE-ready matplotlib rcParams. Call once per script."""
    mpl.rcParams.update({
        "font.family":          "serif",
        "font.size":            11,
        "axes.labelsize":       12,
        "axes.titlesize":       12,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "legend.fontsize":      9,
        "legend.framealpha":    0.9,
        "figure.dpi":           150,   # screen preview
        "savefig.dpi":          300,   # saved to disk
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,
        "axes.grid":            True,
        "grid.alpha":           0.3,
        "grid.linestyle":       "--",
        "lines.linewidth":      1.5,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
    })


def save_fig(fig: plt.Figure, name: str, subdir: str = "") -> Path:
    """
    Save figure to benchmarks/results/<subdir>/<name>.

    Creates parent directories automatically.
    Returns the saved path.
    """
    out_dir = _RESULTS_DIR / subdir if subdir else _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def traj_color(trajectory_type: str) -> str:
    """Return a consistent color for a trajectory type."""
    return TRAJ_TYPE_COLORS.get(trajectory_type, COLORS["default"])


# ---------------------------------------------------------------------------
# Reusable plot helpers
# ---------------------------------------------------------------------------

def dual_axis_timeseries(
    t: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    y1_label: str,
    y2_label: str,
    y1_color: str = COLORS["error"],
    y2_color: str = COLORS["voltage"],
    figsize: Tuple[float, float] = DOUBLE_COL,
    title: str = "",
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Time series with a secondary y-axis.

    Returns (fig, ax1, ax2).
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    t_s = t - t[0]  # relative time in seconds

    ax1.plot(t_s, y1, color=y1_color, linewidth=1.5, label=y1_label)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(y1_label, color=y1_color)
    ax1.tick_params(axis="y", labelcolor=y1_color)

    ax2 = ax1.twinx()
    ax2.plot(t_s, y2, color=y2_color, linewidth=1.2, linestyle="--", label=y2_label)
    ax2.set_ylabel(y2_label, color=y2_color)
    ax2.tick_params(axis="y", labelcolor=y2_color)
    ax2.spines["right"].set_visible(True)  # secondary axis needs right spine

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    if title:
        ax1.set_title(title)

    fig.tight_layout()
    return fig, ax1, ax2


def add_voltage_vlines(
    ax: plt.Axes,
    t: np.ndarray,
    vbat: np.ndarray,
    thresholds: list = [3.8],
    colors: list = ["red"],
) -> None:
    """
    Add vertical dashed lines at timestamps where battery first drops below thresholds.
    """
    t_s = t - t[0]
    for thresh, col in zip(thresholds, colors):
        idx = np.where(vbat < thresh)[0]
        if len(idx) > 0:
            ax.axvline(t_s[idx[0]], color=col, linestyle="--", alpha=0.7,
                       linewidth=1.0, label=f"Vbat < {thresh}V")
