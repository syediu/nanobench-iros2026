# -*- coding: utf-8 -*-
"""
data_loader.py — Discover and load NanoBench aligned.csv datasets.

Searches multiple candidate dataset root directories and returns a list of
TrajectoryData objects with verified column names and derived metadata.
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Root of the nanobench package (3 levels up from this file:
# benchmarks/utils/data_loader.py -> benchmarks/utils -> benchmarks -> nanobench)
_NANOBENCH_ROOT = Path(__file__).resolve().parent.parent.parent

# Candidate dataset root directories to search (in order)
_CANDIDATE_ROOTS = [
    _NANOBENCH_ROOT / "datasets",
    _NANOBENCH_ROOT / ".." / ".." / "datasets",
    _NANOBENCH_ROOT / ".." / "crazyswarm" / "scripts" / "nanobench_dataset",
]

MOTOR_COLS = ["motor_motor_m1", "motor_motor_m2", "motor_motor_m3", "motor_motor_m4"]
REQUIRED_COLS = ["t", "px", "py", "pz", "pwr_pm_vbat"] + MOTOR_COLS

CRAZYFLIE_MASS_KG = 0.027  # from experiment.yaml; overridden by metadata if present


@dataclass
class TrajectoryData:
    path: str                           # directory containing aligned.csv
    traj_id: str                        # e.g. "B2_circle_slow_rep01"
    trajectory_type: str                # circle, figure_eight, hover, etc.
    category: str                       # tracking, excitation, calibration
    repetition: int
    df: pd.DataFrame                    # full aligned.csv (all rows, including pre/post-flight)
    metadata: Dict[str, Any]
    mean_voltage: float
    min_voltage: float
    max_voltage: float
    n_samples: int                      # total rows in df
    n_flight_samples: int               # rows where motors are active
    mass_kg: float = CRAZYFLIE_MASS_KG

    @property
    def flight_mask(self) -> np.ndarray:
        """Boolean mask selecting active-flight rows (motor_m1 > 1000)."""
        return (self.df["motor_motor_m1"] > 1000).values

    @property
    def flight_df(self) -> pd.DataFrame:
        """Subset of df during active flight only."""
        return self.df[self.flight_mask].copy()

    def __repr__(self) -> str:
        return (f"TrajectoryData({self.traj_id!r}, type={self.trajectory_type!r}, "
                f"n={self.n_samples}, flight={self.n_flight_samples}, "
                f"V=[{self.min_voltage:.2f},{self.max_voltage:.2f}])")


def _load_metadata(meta_path: Path) -> Dict[str, Any]:
    """Load metadata.yaml; return empty dict on failure."""
    try:
        with open(meta_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Could not load metadata from {meta_path}: {e}")
        return {}


def _reconstruct_att_qw(df: pd.DataFrame) -> pd.DataFrame:
    """Add att_qw column. Use att_stateEstimate_qw if present (from logging), else reconstruct from qx,qy,qz."""
    df = df.copy()
    if "att_stateEstimate_qw" in df.columns:
        df["att_qw"] = df["att_stateEstimate_qw"].values
        return df
    qx = df["att_stateEstimate_qx"].values
    qy = df["att_stateEstimate_qy"].values
    qz = df["att_stateEstimate_qz"].values
    qw = np.sqrt(np.maximum(0.0, 1.0 - qx**2 - qy**2 - qz**2))
    df["att_qw"] = qw
    return df


def _load_aligned_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load aligned.csv; return None on failure or missing required columns."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"Failed to read {csv_path}: {e}")
        return None

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.warning(f"{csv_path}: missing required columns {missing}, skipping")
        return None

    # Forward-fill battery voltage (10 Hz power log → step-wise values at 200 Hz)
    df["pwr_pm_vbat"] = df["pwr_pm_vbat"].replace(0.0, np.nan).ffill().bfill()

    df = _reconstruct_att_qw(df)
    return df


def _parse_traj_id(directory_name: str) -> str:
    """Extract trajectory ID from directory name, e.g. 'B2_circle_slow_rep01'."""
    return directory_name


def load_trajectory(traj_dir: Path) -> Optional[TrajectoryData]:
    """
    Load a single trajectory directory.

    Returns None if the directory is invalid, has insufficient samples,
    or all motor commands are zero.
    """
    csv_path = traj_dir / "aligned.csv"
    if not csv_path.exists():
        return None

    df = _load_aligned_csv(csv_path)
    if df is None:
        return None

    if len(df) < 500:
        logger.warning(f"{traj_dir.name}: only {len(df)} rows, skipping (< 500)")
        return None

    flight_mask = df["motor_motor_m1"] > 1000
    n_flight = int(flight_mask.sum())
    if n_flight < 100:
        logger.warning(f"{traj_dir.name}: only {n_flight} flight rows, skipping")
        return None

    metadata = _load_metadata(traj_dir / "metadata.yaml")

    traj_id = _parse_traj_id(traj_dir.name)
    trajectory_type = metadata.get("trajectory_type", "unknown")
    category = metadata.get("category", "unknown")
    repetition = metadata.get("repetition", 1)

    # Mass: prefer metadata field if available
    mass_kg = CRAZYFLIE_MASS_KG
    if "mass_grams" in metadata:
        mass_kg = float(metadata["mass_grams"]) / 1000.0

    vbat = df["pwr_pm_vbat"]
    mean_v = float(vbat.mean())
    min_v = float(vbat.min())
    max_v = float(vbat.max())

    return TrajectoryData(
        path=str(traj_dir),
        traj_id=traj_id,
        trajectory_type=trajectory_type,
        category=category,
        repetition=repetition,
        df=df,
        metadata=metadata,
        mean_voltage=mean_v,
        min_voltage=min_v,
        max_voltage=max_v,
        n_samples=len(df),
        n_flight_samples=n_flight,
        mass_kg=mass_kg,
    )


def discover_datasets(base_dirs: Optional[List[Path]] = None) -> List[TrajectoryData]:
    """
    Walk candidate dataset directories and return all valid TrajectoryData objects.

    Parameters
    ----------
    base_dirs : list of Path, optional
        Override the default search locations.

    Returns
    -------
    List[TrajectoryData] sorted by (trajectory_type, traj_id).
    """
    if base_dirs is None:
        search_roots = [p.resolve() for p in _CANDIDATE_ROOTS]
    else:
        search_roots = [Path(p).resolve() for p in base_dirs]

    trajectories: List[TrajectoryData] = []
    searched_dirs: List[Path] = []

    for root in search_roots:
        if not root.exists():
            logger.debug(f"Skipping non-existent dataset root: {root}")
            continue

        logger.info(f"Searching for datasets in: {root}")
        searched_dirs.append(root)

        # Walk the tree looking for directories that contain aligned.csv
        for csv_file in root.rglob("aligned.csv"):
            traj_dir = csv_file.parent
            traj = load_trajectory(traj_dir)
            if traj is not None:
                trajectories.append(traj)
                logger.info(f"  Loaded: {traj}")
            else:
                logger.debug(f"  Skipped: {traj_dir.name}")

    if not trajectories:
        warnings.warn(
            f"No valid trajectories found. Searched: {searched_dirs}",
            RuntimeWarning,
        )
    else:
        logger.info(f"Total trajectories loaded: {len(trajectories)}")

    # Sort for reproducibility
    trajectories.sort(key=lambda t: (t.trajectory_type, t.traj_id))
    return trajectories


def get_best_trajectory(trajectories: List[TrajectoryData],
                        prefer_type: str = "circle") -> Optional[TrajectoryData]:
    """
    Return the trajectory with the largest voltage range (best for voltage analysis),
    optionally preferring a specific trajectory type.
    """
    if not trajectories:
        return None
    # Prefer the requested type if available
    typed = [t for t in trajectories if t.trajectory_type == prefer_type]
    pool = typed if typed else trajectories
    return max(pool, key=lambda t: t.max_voltage - t.min_voltage)
