#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 2 (Controller Benchmarking) — Offline baselines.

Implements three controller baselines trained OFFLINE from aligned.csv logs:
  1) BC-MLP (imitation)
  2) BC-LSTM (custom PyTorch)
  3) MPPI (pytorch-mppi) with learned dynamics

All evaluation is closed-loop rollout through a learned dynamics model trained from logs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Ensure repository root is importable when invoked as a script.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent.parent  # .../nanobench
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.task2_control.data.loader import (
    ACT_COLS,
    OBS_COLS,
    SplitMetadata,
    discover_aligned_csvs,
    load_transitions,
    save_split_metadata,
    split_by_file,
    MOTOR_MAX_PWM,
)
from benchmarks.task2_control.baselines.bc_mlp import BCMLPConfig, BCMLPController, train_bc_mlp
from benchmarks.task2_control.baselines.bc_lstm import (
    BCLSTMConfig,
    BCLSTMController,
    LSTMController,
    train_bc_lstm,
)
from benchmarks.task2_control.baselines.mppi_controller import (
    MPPIConfig,
    MPPIController,
    make_running_cost_with_bc_prior,
)
from benchmarks.task2_control.dynamics.learned_dynamics import (
    DynamicsMLPConfig,
    load_dynamics,
    save_dynamics,
    train_dynamics_mlp,
)
from benchmarks.task2_control.evaluation.rollout import rollout_controller
from benchmarks.task2_control.evaluation.metrics import BATTERY_BINS, compute_metrics
from benchmarks.task2_control.visualization.plots import (
    _out_dirs,
    plot_training_curves,
    plot_open_loop_motor_predictions,
    plot_closed_loop_xyz,
    plot_rmse_vs_battery,
    plot_rmse_vs_trajectory_type,
    plot_3d_trajectory,
    save_pdf_png,
    save_rep_trajectory_plot_data,
    save_tables_latex_csv,
)


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Task2Config:
    seed: int = 42
    dataset_root: str = "datasets/dataset"
    out_dir: str = "benchmarks/task2_control/results"
    dry_run: bool = False


def _print_config(cfg: Task2Config) -> None:
    print("=== Task 2 config ===")
    for k, v in asdict(cfg).items():
        print(f"{k:>16s}: {v}")


def _load_obs_actions_for_file(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load unscaled obs (T,14), expert acts (T,4 in [0,1]), and time (T,)."""
    from benchmarks.task2_control.data.loader import (
        _reconstruct_att_qw,
        VBAT_MIN_V,
        VBAT_MAX_V,
        SP_Z_MIN_M,
        SP_Z_MAX_M,
        PWM_MIN,
        PWM_MAX,
    )

    df = pd.read_csv(csv_path)
    df = _reconstruct_att_qw(df)
    missing = [c for c in OBS_COLS + ACT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {missing}")
    df["pwr_pm_vbat"] = df["pwr_pm_vbat"].replace(0.0, np.nan).ffill().bfill()
    df = df[df["motor_motor_m1"] > 1000].reset_index(drop=True)

    q_lo, q_hi = 0.5, 99.5
    v = df["pwr_pm_vbat"].to_numpy(float)
    v_lo, v_hi = float(np.nanpercentile(v, q_lo)), float(np.nanpercentile(v, q_hi))
    if (v_lo < VBAT_MIN_V) or (v_hi > VBAT_MAX_V):
        raise ValueError(f"{csv_path}: vbat out of range p{q_lo}={v_lo:.2f}, p{q_hi}={v_hi:.2f} V")
    if "sp_ctrltarget_z" in df.columns:
        z = df["sp_ctrltarget_z"].to_numpy(float)
        z_lo = float(np.nanpercentile(z, q_lo))
        z_hi = float(np.nanpercentile(z, q_hi))
        if (z_lo < SP_Z_MIN_M) or (z_hi > SP_Z_MAX_M):
            raise ValueError(f"{csv_path}: sp_z out of range p{q_lo}={z_lo:.2f}, p{q_hi}={z_hi:.2f} m")
    u_pwm = df[ACT_COLS].to_numpy(float)
    u_lo = float(np.nanpercentile(u_pwm, q_lo))
    u_hi = float(np.nanpercentile(u_pwm, q_hi))
    if (u_lo < PWM_MIN) or (u_hi > PWM_MAX):
        raise ValueError(f"{csv_path}: motor pwm out of range p{q_lo}={u_lo:.1f}, p{q_hi}={u_hi:.1f}")

    t = df["t"].to_numpy(float) if "t" in df.columns else np.arange(len(df), dtype=float)
    obs = df[OBS_COLS].to_numpy(np.float32)
    acts = (df[ACT_COLS].to_numpy(np.float32) / float(MOTOR_MAX_PWM))
    acts = np.clip(acts, 0.0, 1.0)
    return obs, acts, t


def _trajectory_type_for_run(csv_path: Path) -> str:
    """Read trajectory_type from metadata.yaml when available.

    Supports two layouts:
      - Nested: .../traj_dir/aligned.csv → traj_dir/metadata.yaml
      - Flat:   .../B3_figure8_medium_rep1.csv → B3_figure8_medium_rep1_metadata.yaml
    """
    meta = csv_path.parent / "metadata.yaml"
    if not meta.exists():
        meta = csv_path.parent / (csv_path.stem + "_metadata.yaml")
    if not meta.exists():
        return "unknown"
    try:
        import yaml
        with open(meta, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return str(d.get("trajectory_type", "unknown"))
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="datasets/dataset")
    ap.add_argument("--out-dir", default=str(Path("benchmarks/task2_control/results").resolve()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="2 epochs, small data, skip heavy plots.")
    ap.add_argument("--eval-only", "--eval_only", dest="eval_only", action="store_true",
                    help="Skip training; load checkpoints and evaluate.")
    ap.add_argument("--retrain-dynamics", action="store_true", help="Retrain dynamics model (keeps BC policies).")
    args = ap.parse_args()

    cfg = Task2Config(seed=args.seed, dataset_root=args.dataset_root, out_dir=args.out_dir, dry_run=args.dry_run)
    _print_config(cfg)
    _set_seeds(cfg.seed)

    task_root = Path(__file__).resolve().parent
    fig_dir, tab_dir = _out_dirs(task_root)
    ckpt_dir = task_root / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(cfg.dataset_root).resolve()
    files = discover_aligned_csvs(dataset_root)
    if not files:
        raise SystemExit(f"No aligned.csv found under {dataset_root}")

    train_files, val_files, test_files = split_by_file(files, seed=cfg.seed)
    split_meta = SplitMetadata(
        seed=cfg.seed,
        train_files=[str(p) for p in train_files],
        val_files=[str(p) for p in val_files],
        test_files=[str(p) for p in test_files],
    )
    save_split_metadata(task_root / "results" / "split_metadata.json", split_meta)

    if cfg.dry_run:
        train_files = train_files[:2]
        val_files = val_files[:1] if val_files else train_files[:1]
        test_files = test_files[:1] if test_files else train_files[:1]

    # Load transitions + scaler
    scaler = None
    if args.eval_only:
        import joblib
        scaler = joblib.load(ckpt_dir / "obs_scaler.pkl")
        trans_train, _ = load_transitions(train_files[:1], scaler=scaler, fit_scaler=False)
        trans_val, _ = load_transitions(val_files[:1] if val_files else train_files[:1], scaler=scaler, fit_scaler=False)
    else:
        trans_train, scaler = load_transitions(train_files, scaler=None, fit_scaler=True)
        trans_val, _ = load_transitions(val_files or train_files[:1], scaler=scaler, fit_scaler=False)

    try:
        import joblib
        joblib.dump(scaler, ckpt_dir / "obs_scaler.pkl")
    except Exception as e:
        print(f"WARNING: failed to save scaler: {e}")

    if cfg.dry_run:
        max_n = 200
        trans_train = type(trans_train)(
            obs=trans_train.obs[:max_n], acts=trans_train.acts[:max_n],
            next_obs=trans_train.next_obs[:max_n], dones=trans_train.dones[:max_n],
            infos=trans_train.infos[:max_n],
        )
        trans_val = type(trans_val)(
            obs=trans_val.obs[:max_n], acts=trans_val.acts[:max_n],
            next_obs=trans_val.next_obs[:max_n], dones=trans_val.dones[:max_n],
            infos=trans_val.infos[:max_n],
        )

    # BC-MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bc_train_s = float("nan")
    if args.eval_only:
        bc_policy = torch.load(ckpt_dir / "bc_mlp_policy.pt", map_location="cpu", weights_only=False)
    else:
        bc_cfg = BCMLPConfig(seed=cfg.seed, n_epochs=2 if cfg.dry_run else 600,
                             batch_size=64 if cfg.dry_run else 256)
        import time
        t0 = time.time()
        _, bc_policy = train_bc_mlp(trans_train, cfg=bc_cfg, device=str(device))
        bc_train_s = time.time() - t0
        try:
            torch.save(bc_policy, ckpt_dir / "bc_mlp_policy.pt")
        except Exception as e:
            print(f"WARNING: failed to save BC-MLP policy: {e}")
    bc_controller = BCMLPController(bc_policy, device=device)

    # BC-LSTM
    lstm_cfg = BCLSTMConfig(seed=cfg.seed, max_epochs=2 if cfg.dry_run else 600, patience=80)

    def _files_to_lists(file_list: List[Path]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        obs_list, act_list, t_list = [], [], []
        for p in file_list:
            try:
                obs, act, t = _load_obs_actions_for_file(p)
            except Exception:
                continue
            if len(obs) < 5:
                continue
            obs_s = scaler.transform(obs).astype(np.float32)
            obs_list.append(obs_s)
            act_list.append(act.astype(np.float32))
            t_list.append(t.astype(np.float64))
        return obs_list, act_list, t_list

    tr_obs_list, tr_act_list, _ = _files_to_lists(train_files)
    va_obs_list, va_act_list, _ = _files_to_lists(val_files or train_files[:1])
    lstm_train_s = float("nan")
    if args.eval_only:
        ckpt = torch.load(ckpt_dir / "bc_lstm.pt", map_location="cpu", weights_only=False)
        lstm_model = LSTMController().to(device)
        lstm_model.load_state_dict(ckpt["model_state"])
        lstm_model.eval()
        lstm_hist = {"train_loss": [], "val_loss": []}
    else:
        import time
        t0 = time.time()
        lstm_model, lstm_hist = train_bc_lstm(tr_obs_list, tr_act_list, va_obs_list, va_act_list, cfg=lstm_cfg)
        lstm_train_s = time.time() - t0
    lstm_controller = BCLSTMController(lstm_model, seq_len=lstm_cfg.seq_len, device=device)
    try:
        torch.save({"model_state": lstm_model.state_dict(), "config": asdict(lstm_cfg)}, ckpt_dir / "bc_lstm.pt")
    except Exception as e:
        print(f"WARNING: failed to save BC-LSTM checkpoint: {e}")

    # Dynamics MLP
    dyn_train_s = float("nan")
    dyn_hist = {"train_loss": [], "val_loss": []}
    dynamics_predicts_delta = False
    if args.eval_only and not args.retrain_dynamics:
        dyn_model, extra = load_dynamics(ckpt_dir / "dynamics_mlp.pt", device=device)
        dynamics_predicts_delta = bool(extra.get("predict_delta", False))
    else:
        dyn_cfg = DynamicsMLPConfig(
            seed=cfg.seed,
            max_epochs=2 if cfg.dry_run else 600,
            patience=80,
            predict_delta=True,
            action_noise_std=0.02,
        )
        import time
        t0 = time.time()
        dyn_model, dyn_hist = train_dynamics_mlp(trans_train, trans_val, cfg=dyn_cfg)
        dyn_train_s = time.time() - t0
        dynamics_predicts_delta = True
        try:
            save_dynamics(dyn_model, ckpt_dir / "dynamics_mlp.pt",
                          extra={"predict_delta": True, "config": asdict(dyn_cfg)})
        except Exception as e:
            print(f"WARNING: failed to save dynamics checkpoint: {e}")

    # MPPI
    obs_mean = scaler.mean_.astype(np.float32)
    obs_scale = scaler.scale_.astype(np.float32)
    running_cost, step_counter = make_running_cost_with_bc_prior(
        obs_mean=obs_mean, obs_scale=obs_scale, bc_controller=bc_controller,
        pos_weight=1.0, vel_weight=0.05, action_weight=0.01, prior_weight=2.0,
    )

    @torch.no_grad()
    def dynamics_fn(state_scaled: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pred = dyn_model(state_scaled, action)
        return state_scaled + pred if dynamics_predicts_delta else pred

    mppi_cfg = MPPIConfig(seed=cfg.seed, horizon=10, num_samples=128, noise_sigma=0.02)
    mppi_controller = MPPIController(
        dynamics_fn=dynamics_fn, running_cost_fn=running_cost, step_counter=step_counter,
        obs_dim=14, cfg=mppi_cfg, device=device,
    )

    controllers = {"BC-MLP": bc_controller, "BC-LSTM": lstm_controller, "MPPI": mppi_controller}

    # Open-loop stats (BC only)
    open_loop_abs_err: Dict[str, List[np.ndarray]] = {"BC-MLP": [], "BC-LSTM": []}
    open_loop_mse: Dict[str, List[float]] = {"BC-MLP": [], "BC-LSTM": []}
    for test_path in test_files:
        try:
            obs_u, act_u, _ = _load_obs_actions_for_file(test_path)
        except Exception as e:
            print(f"WARNING: skipping open-loop test stats for {test_path.parent.name}: {e}")
            continue
        exp_actions = act_u[:-1]
        for name in ["BC-MLP", "BC-LSTM"]:
            ctrl = controllers[name]
            preds = []
            for k in range(len(exp_actions)):
                o_scaled = scaler.transform(obs_u[k : k + 1]).astype(np.float32)[0]
                preds.append(ctrl.predict(o_scaled))
            uhat = np.asarray(preds, dtype=np.float32)
            err = uhat - exp_actions
            open_loop_abs_err[name].append(np.abs(err))
            open_loop_mse[name].append(float(np.mean(err * err)))

    all_rows: List[Dict] = []
    rmse_by_model_and_bin = {k: {b.name: [] for b in BATTERY_BINS} for k in controllers}
    rmse_by_model_and_type = {k: {} for k in controllers}
    rep_obs_u: Optional[np.ndarray] = None
    rep_exp_actions: Optional[np.ndarray] = None
    rep_t: Optional[np.ndarray] = None
    rep_trajectory_name: Optional[str] = None

    for idx, test_path in enumerate(test_files):
        print(f"[Eval] Trajectory {idx+1}/{len(test_files)}: {test_path.parent.name}")
        try:
            obs_u, act_u, t = _load_obs_actions_for_file(test_path)
        except Exception as e:
            print(f"WARNING: skipping eval for {test_path.parent.name}: {e}")
            continue
        exp_actions = act_u[:-1]
        traj_type = _trajectory_type_for_run(test_path)
        if rep_obs_u is None:
            rep_obs_u = obs_u.copy()
            rep_exp_actions = exp_actions.copy()
            rep_t = t.copy()
            rep_trajectory_name = test_path.parent.name

        for name, ctrl in controllers.items():
            if name == "MPPI":
                step_counter[0] = 0
            print(f"  [Eval]   {name} rollout...")
            res = rollout_controller(
                controller=ctrl,
                dynamics_model=dyn_model,
                test_obs=obs_u,
                test_actions_expert=exp_actions,
                obs_scaler=scaler,
                device=device,
                override_exogenous=True,
                dynamics_predicts_delta=dynamics_predicts_delta,
            )
            m = compute_metrics(res.rollout_obs, res.gt_obs, res.actions_pred, res.actions_expert)
            row = {
                "trajectory": str(test_path.parent.name),
                "trajectory_type": traj_type,
                "model": name,
                **m,
                "mean_vbat_V": float(np.mean(obs_u[:, 13])),
                "min_vbat_V": float(np.min(obs_u[:, 13])),
            }
            all_rows.append(row)
            vmean = row["mean_vbat_V"]
            for b in BATTERY_BINS:
                if b.lo <= vmean < b.hi:
                    rmse_by_model_and_bin[name][b.name].append(float(m["pos_RMSE_m"]))
            rmse_by_model_and_type[name].setdefault(traj_type, []).append(float(m["pos_RMSE_m"]))

    summary_path = task_root / "results" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": asdict(cfg),
            "checkpoints": {
                "obs_scaler": str((ckpt_dir / "obs_scaler.pkl").resolve()),
                "bc_mlp_policy": str((ckpt_dir / "bc_mlp_policy.pt").resolve()),
                "bc_lstm": str((ckpt_dir / "bc_lstm.pt").resolve()),
                "dynamics_mlp": str((ckpt_dir / "dynamics_mlp.pt").resolve()),
            },
            "rows": all_rows,
        }, f, indent=2)
    print(f"Wrote: {summary_path}")

    save_tables_latex_csv(tab_dir, "task2_closed_loop_metrics_per_traj", all_rows)

    t1_rows = []
    for model_name in ["BC-MLP", "BC-LSTM"]:
        if not open_loop_abs_err[model_name]:
            continue
        abs_err = np.concatenate(open_loop_abs_err[model_name], axis=0)
        per_motor_mae = np.mean(abs_err, axis=0)
        action_mse = float(np.mean(open_loop_mse[model_name]))
        t_train = bc_train_s if model_name == "BC-MLP" else lstm_train_s
        t1_rows.append({"model": model_name, "action_MSE": action_mse,
                        "m1_MAE": float(per_motor_mae[0]), "m2_MAE": float(per_motor_mae[1]),
                        "m3_MAE": float(per_motor_mae[2]), "m4_MAE": float(per_motor_mae[3]),
                        "train_time_s": t_train})
    save_tables_latex_csv(tab_dir, "table_task2_open_loop_imitation", t1_rows)

    metrics_cols = ["pos_RMSE_m", "pos_ADE_m", "pos_FDE_m", "vel_RMSE_mps", "heading_error_deg", "divergence_rate_pct"]
    t2_rows = []
    for model_name in controllers:
        rows_m = [r for r in all_rows if r["model"] == model_name]
        if not rows_m:
            continue
        out = {"model": model_name}
        for c in metrics_cols:
            vals = np.array([r[c] for r in rows_m], dtype=float)
            out[c] = float(np.mean(vals))
            out[c + "_std"] = float(np.std(vals))
        t2_rows.append(out)
    save_tables_latex_csv(tab_dir, "table_task2_closed_loop_main", t2_rows)

    t3_rows = []
    for model_name in controllers:
        for tt in sorted({r.get("trajectory_type", "unknown") for r in all_rows}):
            rows_mt = [r for r in all_rows if r["model"] == model_name and r["trajectory_type"] == tt]
            if not rows_mt:
                continue
            t3_rows.append({"model": model_name, "trajectory_type": tt,
                            "pos_RMSE_m": float(np.mean([r["pos_RMSE_m"] for r in rows_mt])),
                            "pos_ADE_m": float(np.mean([r["pos_ADE_m"] for r in rows_mt])),
                            "n_traj": len(rows_mt)})
    save_tables_latex_csv(tab_dir, "table_task2_per_trajectory_type", t3_rows)

    histories = {"BC-LSTM": lstm_hist, "Dyn-MLP": dyn_hist}
    fig = plot_training_curves(histories)
    save_pdf_png(fig, fig_dir / "fig_task2_training_curves")

    if rep_obs_u is not None and rep_exp_actions is not None and rep_t is not None:
        t_s = rep_t - rep_t[0]
        motors_gt = rep_exp_actions
        motors_pred = {}
        for name, ctrl in {"BC-MLP": bc_controller, "BC-LSTM": lstm_controller}.items():
            preds = []
            for k in range(len(motors_gt)):
                o_scaled = scaler.transform(rep_obs_u[k : k + 1]).astype(np.float32)[0]
                preds.append(ctrl.predict(o_scaled))
            motors_pred[name] = np.asarray(preds, dtype=np.float32)
        fig = plot_open_loop_motor_predictions(t_s[: len(motors_gt)], motors_gt, motors_pred)
        save_pdf_png(fig, fig_dir / "fig_task2_open_loop_motor")

        pos_gt = rep_obs_u[:, 0:3]
        pos_rollouts = {}
        for name, ctrl in controllers.items():
            if name == "MPPI":
                step_counter[0] = 0
            rr = rollout_controller(
                ctrl, dyn_model, rep_obs_u, motors_gt, scaler, device,
                override_exogenous=True,
                dynamics_predicts_delta=dynamics_predicts_delta,
            )
            pos_rollouts[name] = rr.rollout_obs[:, 0:3]
        fig = plot_closed_loop_xyz(t_s, pos_gt, pos_rollouts)
        save_pdf_png(fig, fig_dir / "fig_task2_closed_loop_xyz")
        fig = plot_3d_trajectory(pos_gt, pos_rollouts)
        save_pdf_png(fig, fig_dir / "fig_task2_3d_trajectory")

        plot_data_dir = task_root / "results" / "plot_data"
        save_rep_trajectory_plot_data(plot_data_dir, t_s, motors_gt, motors_pred, pos_gt, pos_rollouts,
                                      trajectory_name=rep_trajectory_name or "rep")
        print(f"Wrote plot data CSVs to {plot_data_dir}")
    else:
        print("WARNING: no valid representative trajectory for qualitative plots; skipping motor/XYZ/3D figures.")

    fig = plot_rmse_vs_battery(rmse_by_model_and_bin)
    save_pdf_png(fig, fig_dir / "fig_task2_rmse_vs_battery")
    fig = plot_rmse_vs_trajectory_type(rmse_by_model_and_type)
    save_pdf_png(fig, fig_dir / "fig_task2_rmse_vs_trajectory_type")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
