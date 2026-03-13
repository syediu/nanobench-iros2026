#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone sanity check for the learned dynamics model.
Run this BEFORE the full rollout to verify the model is numerically stable.

Usage:
    python benchmarks/task2_control/debug/sanity_check_dynamics.py \
        --dynamics_path <path_to_dynamics_checkpoint> \
        --scaler_path <path_to_obs_scaler.pkl> \
        --data_path <path_to_one_aligned_csv>

This script does not use controllers or the rollout engine.
"""

# OUTPUT (current checkpoint on B3_figure8_medium_rep01):
# One-step pos error:  mean=0.0066 m, max=0.0145 m
# Rollout pos error at  50 steps: 0.1931 m
# Rollout pos error at 100 steps: 0.5326 m
# Rollout pos error at 200 steps: 0.4773 m

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import sys

# Ensure repo root is importable when invoked as a script
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent.parent.parent  # .../nanobench
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
MOTOR_MAX_PWM = 65535.0


def load_obs_actions(path: Path, n: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "att_stateEstimate_qw" not in df.columns and all(
        c in df.columns for c in ("att_stateEstimate_qx", "att_stateEstimate_qy", "att_stateEstimate_qz")
    ):
        qx = df["att_stateEstimate_qx"].to_numpy(float)
        qy = df["att_stateEstimate_qy"].to_numpy(float)
        qz = df["att_stateEstimate_qz"].to_numpy(float)
        df["att_stateEstimate_qw"] = np.sqrt(np.maximum(0.0, 1.0 - qx * qx - qy * qy - qz * qz))

    df["pwr_pm_vbat"] = df["pwr_pm_vbat"].replace(0.0, np.nan).ffill().bfill()
    df = df[df["motor_motor_m1"] > 1000].reset_index(drop=True)
    df = df.iloc[:n].reset_index(drop=True)

    obs = df[OBS_COLS].to_numpy(np.float32)
    acts = (df[ACT_COLS].to_numpy(np.float32) / float(MOTOR_MAX_PWM))
    acts = np.clip(acts, 0.0, 1.0)
    return obs, acts


def pos_err_m(obs_a: np.ndarray, obs_b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(obs_a[:, 0:3] - obs_b[:, 0:3], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics_path", required=True)
    ap.add_argument("--scaler_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dyn_path = Path(args.dynamics_path).resolve()
    scaler_path = Path(args.scaler_path).resolve()
    data_path = Path(args.data_path).resolve()

    import joblib

    scaler = joblib.load(scaler_path)

    # load dynamics checkpoint (expects {"model_state": ...} or {"model_state":..., "config":...})
    ckpt = torch.load(dyn_path, map_location="cpu")
    state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt.get("model")
    extra = ckpt.get("extra", {}) if isinstance(ckpt, dict) else {}
    predict_delta = bool(extra.get("predict_delta", False))

    # Infer architecture from checkpoint if present
    from benchmarks.task2_control.dynamics.learned_dynamics import MLPDynamics

    model = MLPDynamics().to(args.device)
    model.load_state_dict(state)
    model.eval()

    obs, acts = load_obs_actions(data_path, n=500)
    obs_s = scaler.transform(obs).astype(np.float32)

    # one-step errors (teacher-forced)
    with torch.no_grad():
        x = torch.from_numpy(obs_s[:-1]).to(args.device)
        u = torch.from_numpy(acts[:-1]).to(args.device)
        pred_s = model(x, u).detach().cpu().numpy().astype(np.float32)
        yhat_s = (obs_s[:-1] + pred_s) if predict_delta else pred_s
    yhat = scaler.inverse_transform(yhat_s).astype(np.float32)
    gt_next = obs[1:]

    e1 = pos_err_m(yhat, gt_next)
    print(f"One-step pos error:  mean={float(np.mean(e1)):.4f} m, max={float(np.max(e1)):.4f} m")

    # autoregressive rollout horizons
    horizons = [50, 100, 200]
    errs = {}
    for H in horizons:
        H = min(H, len(obs) - 1)
        roll = np.zeros((H + 1, obs.shape[1]), dtype=np.float32)
        roll[0] = obs[0]
        for t in range(H):
            xs = scaler.transform(roll[t : t + 1]).astype(np.float32)
            with torch.no_grad():
                pred_s = model(
                    torch.from_numpy(xs).to(args.device),
                    torch.from_numpy(acts[t : t + 1]).to(args.device),
                ).detach().cpu().numpy().astype(np.float32)
            y_s = (xs + pred_s) if predict_delta else pred_s
            roll[t + 1] = scaler.inverse_transform(y_s).astype(np.float32)[0]
            # keep exogenous realistic (same assumption as main rollout)
            roll[t + 1, 10:13] = obs[t + 1, 10:13]
            roll[t + 1, 13] = obs[t + 1, 13]

        e = pos_err_m(roll[1:], obs[1 : H + 1])
        errs[H] = float(e[-1])
        print(f"Rollout pos error at {H:3d} steps: {e[-1]:.4g} m")

        if H == 50:
            if e[-1] > 1.0:
                idx = int(np.argmax(e > 0.1)) if np.any(e > 0.1) else -1
                print("WARNING: dynamics model is unstable")
                if idx >= 0:
                    print(f"First step where pos error exceeds 0.1 m: step {idx}")

    # plot error vs horizon
    hs = np.array(sorted(errs.keys()), dtype=int)
    ys = np.array([errs[h] for h in hs], dtype=float)
    out_png = data_path.parent / "rollout_stability.png"
    plt.figure(figsize=(4.5, 3.0))
    plt.plot(hs, ys, "o-", linewidth=1.5)
    plt.xlabel("Rollout horizon [steps]")
    plt.ylabel("Position error at horizon [m]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

