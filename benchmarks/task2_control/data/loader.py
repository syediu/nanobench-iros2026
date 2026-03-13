"""aligned.csv -> offline imitation learning datasets (Task 2).

The "expert" is the logged Crazyflie PID: motor PWM commands in aligned.csv.

We construct:
  obs  (N,14) = [px,py,pz,
                est_vx,est_vy,est_vz,
                att_qw,att_qx,att_qy,att_qz,
                sp_x,sp_y,sp_z,
                vbat]
  acts (N,4)  = [m1,m2,m3,m4]/65535 in [0,1]

Transitions are built per-trajectory and concatenated; we never mix samples
across trajectory boundaries (no within-trajectory splitting).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from imitation.data.types import Transitions
except Exception as e:  # pragma: no cover
    Transitions = None  # type: ignore
    _IMITATION_IMPORT_ERROR = e


MOTOR_MAX_PWM = 65535.0

OBS_DIM = 14
ACT_DIM = 4

OBS_COLS = [
    "px",
    "py",
    "pz",
    "est_stateEstimate_vx",
    "est_stateEstimate_vy",
    "est_stateEstimate_vz",
    "att_stateEstimate_qw",
    "att_stateEstimate_qx",
    "att_stateEstimate_qy",
    "att_stateEstimate_qz",
    "sp_ctrltarget_x",
    "sp_ctrltarget_y",
    "sp_ctrltarget_z",
    "pwr_pm_vbat",
]

ACT_COLS = ["motor_motor_m1", "motor_motor_m2", "motor_motor_m3", "motor_motor_m4"]

# Plausibility checks (match NanoBench analysis conventions)
VBAT_MIN_V = 2.5
VBAT_MAX_V = 4.5
SP_Z_MIN_M = -0.2
SP_Z_MAX_M = 3.0
PWM_MIN = 0.0
PWM_MAX = 65535.0


@dataclass(frozen=True)
class SplitMetadata:
    seed: int
    train_files: List[str]
    val_files: List[str]
    test_files: List[str]


def _ensure_imitation_available() -> None:
    if Transitions is None:  # pragma: no cover
        raise ImportError(
            "Failed to import imitation. Install with `pip install imitation`.\n"
            f"Original error: {_IMITATION_IMPORT_ERROR}"
        )


def discover_aligned_csvs(dataset_root: str | Path) -> List[Path]:
    """Return trajectory CSV files from dataset_root.

    Supports two layouts:
      - Nested (nanobench_v1): exp_*/traj_dir/aligned.csv
      - Flat (datasets/dataset): <traj_id>_rep<N>.csv
    """
    root = Path(dataset_root).expanduser().resolve()
    # Try nested aligned.csv layout first
    files = sorted(root.rglob("aligned.csv"))
    if files:
        return [p for p in files if p.is_file()]
    # Fall back to flat layout: files named <traj_id>_rep<N>.csv
    import re
    flat = [
        p for p in sorted(root.glob("*.csv"))
        if re.search(r"_rep\d+\.csv$", p.name)
    ]
    return flat


def split_by_file(
    files: Sequence[Path],
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Random split at file boundaries (never within a trajectory)."""
    files = list(files)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n = len(files)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_train = min(max(n_train, 1), n)
    n_val = min(max(n_val, 0), n - n_train)
    train = [files[i] for i in idx[:n_train]]
    val = [files[i] for i in idx[n_train : n_train + n_val]]
    test = [files[i] for i in idx[n_train + n_val :]]
    return train, val, test


def save_split_metadata(path: str | Path, meta: SplitMetadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)


def _reconstruct_att_qw(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure att_stateEstimate_qw exists; reconstruct if missing."""
    if "att_stateEstimate_qw" in df.columns:
        return df
    for c in ("att_stateEstimate_qx", "att_stateEstimate_qy", "att_stateEstimate_qz"):
        if c not in df.columns:
            raise ValueError(f"Missing {c}; cannot reconstruct att_stateEstimate_qw.")
    qx = df["att_stateEstimate_qx"].to_numpy(float)
    qy = df["att_stateEstimate_qy"].to_numpy(float)
    qz = df["att_stateEstimate_qz"].to_numpy(float)
    qw = np.sqrt(np.maximum(0.0, 1.0 - qx * qx - qy * qy - qz * qz))
    out = df.copy()
    out["att_stateEstimate_qw"] = qw
    return out


def _load_one(csv_path: Path, flight_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    df = _reconstruct_att_qw(df)

    missing = [c for c in OBS_COLS + ACT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {missing}")

    # Forward-fill Vbat (power is slower-rate; aligned logs often step-hold)
    df["pwr_pm_vbat"] = df["pwr_pm_vbat"].replace(0.0, np.nan).ffill().bfill()

    if flight_only and "motor_motor_m1" in df.columns:
        mask = (df["motor_motor_m1"].to_numpy(float) > 1000.0)
        df = df.loc[mask].reset_index(drop=True)

    # Basic plausibility checks to exclude corrupted runs (prevents catastrophic rollouts)
    q_lo, q_hi = 0.5, 99.5
    v = df["pwr_pm_vbat"].to_numpy(float)
    v_lo = float(np.nanpercentile(v, q_lo))
    v_hi = float(np.nanpercentile(v, q_hi))
    if (v_lo < VBAT_MIN_V) or (v_hi > VBAT_MAX_V):
        raise ValueError(f"{csv_path}: vbat out of range p{q_lo}={v_lo:.2f}, p{q_hi}={v_hi:.2f} V")
    if "sp_ctrltarget_z" in df.columns:
        z = df["sp_ctrltarget_z"].to_numpy(float)
        z_lo = float(np.nanpercentile(z, q_lo))
        z_hi = float(np.nanpercentile(z, q_hi))
        if (z_lo < SP_Z_MIN_M) or (z_hi > SP_Z_MAX_M):
            raise ValueError(f"{csv_path}: sp_z out of range p{q_lo}={z_lo:.2f}, p{q_hi}={z_hi:.2f} m")
    u = df[ACT_COLS].to_numpy(float)
    u_lo = float(np.nanpercentile(u, q_lo))
    u_hi = float(np.nanpercentile(u, q_hi))
    if (u_lo < PWM_MIN) or (u_hi > PWM_MAX):
        raise ValueError(f"{csv_path}: motor pwm out of range p{q_lo}={u_lo:.1f}, p{q_hi}={u_hi:.1f}")

    obs = df[OBS_COLS].to_numpy(dtype=np.float32, copy=True)
    acts = (df[ACT_COLS].to_numpy(dtype=np.float32, copy=True) / float(MOTOR_MAX_PWM))
    acts = np.clip(acts, 0.0, 1.0)
    return obs, acts


def load_transitions(
    csv_files: Sequence[str | Path],
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
    flight_only: bool = True,
) -> Tuple["Transitions", StandardScaler]:
    """
    Load aligned.csv files and construct imitation Transitions.

    - next_obs is obs shifted by one timestep within each trajectory
    - dones=True only for the final transition of each file
    - obs are standardized with StandardScaler (fit on training if fit_scaler=True)
    """
    _ensure_imitation_available()

    obs_list: List[np.ndarray] = []
    next_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    done_list: List[np.ndarray] = []

    for p in map(Path, csv_files):
        try:
            obs, acts = _load_one(p, flight_only=flight_only)
        except Exception as e:
            # Skip corrupted/malformed trajectories; caller controls dataset size via file list.
            continue
        if len(obs) < 2:
            continue
        # transitions length = N-1
        obs_t = obs[:-1]
        next_t = obs[1:]
        act_t = acts[:-1]
        done = np.zeros((len(obs_t),), dtype=bool)
        done[-1] = True

        obs_list.append(obs_t)
        next_list.append(next_t)
        act_list.append(act_t)
        done_list.append(done)

    if not obs_list:
        raise RuntimeError("No valid transitions loaded (check dataset paths and flight_only mask).")

    obs_all = np.concatenate(obs_list, axis=0)
    next_all = np.concatenate(next_list, axis=0)
    acts_all = np.concatenate(act_list, axis=0)
    dones_all = np.concatenate(done_list, axis=0)

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        scaler.fit(obs_all)

    obs_all_s = scaler.transform(obs_all).astype(np.float32, copy=False)
    next_all_s = scaler.transform(next_all).astype(np.float32, copy=False)

    # imitation expects infos to be a sequence with same length as obs
    infos = [{} for _ in range(len(obs_all_s))]
    trans = Transitions(
        obs=obs_all_s,
        acts=acts_all.astype(np.float32, copy=False),
        next_obs=next_all_s,
        dones=dones_all,
        infos=infos,
    )
    return trans, scaler

