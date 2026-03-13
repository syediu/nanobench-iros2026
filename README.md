# NanoBench: A Multi-Task Benchmark Dataset for Nano-Quadrotor Research

**NanoBench** is an open dataset and reproducible benchmark suite for nano-quadrotor research, collected from a Crazyflie 2.1 platform under Vicon motion-capture ground truth. The dataset spans 170 flight trajectories across 27 trajectory types, totalling approximately 97.5 minutes of synchronized flight data. Three benchmark tasks are provided covering system identification, controller, and onboard state estimation — each with runnable baselines and structured evaluation outputs.

This repository accompanies the following publication:

```bibtex
@misc{ullah2026nanobenchmultitaskbenchmarkdataset,
      title={NanoBench: A Multi-Task Benchmark Dataset for Nano-Quadrotor System Identification, Control, and State Estimation}, 
      author={Syed Izzat Ullah and Jose Baca},
      year={2026},
      eprint={2603.09908},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.09908}, 
}
```

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Environment Setup](#environment-setup)
5. [Task 1 — System Identification](#task-1--system-identification)
6. [Task 2 — Control](#task-2--imitation-learning-control)
7. [Task 3 — State Estimation](#task-3--state-estimation)
8. [License](#license)


---

## Overview

The Crazyflie 2.1 is among the most widely used platforms for nano-quadrotor research, yet publicly available datasets that include synchronized Vicon ground truth, onboard EKF estimates, motor commands, IMU readings, and battery telemetry from a single flight are scarce. NanoBench fills this gap by providing:

- **170 flight recordings** across 27 trajectory types (circles, figure-eights, helices, trefoil knots, lissajous curves, random waypoints, battery drain, and more), each with multiple independent repetitions.
- **51 synchronized signal columns** per recording: Vicon ground-truth position, velocity, and orientation; onboard EKF state estimates; raw IMU (accelerometer + gyroscope); per-motor PWM commands; flight controller setpoints and PID outputs; and battery voltage.
- **Three ready-to-run benchmark tasks** with baseline implementations and standardized metrics, structured to facilitate direct comparison of new methods against established algorithms.

All data are in flat CSV format, readable without any robotics middleware.

---

## Repository Structure

```
nanobench/
├── datasets/
│   └── dataset/                    # 170 flight CSV files + paired metadata YAMLs
│       ├── B2_circle_slow_rep1.csv
│       ├── B2_circle_slow_rep1_metadata.yaml
│       └── ...
│
└── benchmarks/
    ├── utils/
    │   ├── data_loader.py          # Trajectory discovery and loading (TrajectoryData)
    │   ├── metrics.py              # ATE, RTE, RMSE
    │   └── plotting.py             # matplotlib style and save helpers
    │
    ├── task1_sysid/                # Task 1: quadrotor system identification
    │   ├── run_nanobench_sysid.py  # Entry point (setup / train / test / compare)
    │   ├── models/models.py        # Physics, Residual, Phys+Res, LSTM models
    │   ├── dataset/
    │   │   ├── dataset.py          # QuadDataset (PyTorch)
    │   │   └── convert_csvs.py     # NanoBench CSV → IDSIA format converter
    │   ├── train/                  # Training scripts for learned models
    │   ├── test/                   # Inference scripts
    │   └── data/                   # Auto-populated: data/train/, data/test/
    │
    ├── task2_control/              # Task 2: offline imitation-learning controllers
    │   ├── run_task2.py            # Entry point
    │   ├── data/loader.py          # Imitation-learning dataset builder
    │   ├── baselines/
    │   │   ├── bc_mlp.py           # Behavioural Cloning MLP
    │   │   ├── bc_lstm.py          # Behavioural Cloning LSTM
    │   │   └── mppi_controller.py  # MPPI with learned dynamics prior
    │   ├── dynamics/               # Learned forward dynamics model
    │   ├── evaluation/             # Closed-loop rollout evaluator
    │   └── results/                # Auto-populated outputs
    │
    ├── task3_stateEst/             # Task 3: EKF evaluation on trefoil trajectories
        ├── run_task3_trefoil.py    # Entry point
        ├── mellinger_trefoil/      # 15 Mellinger-controller trefoil flights
        ├── pid_trefoil/            # 13 PID-controller trefoil flights
        └── results/                # Auto-populated outputs

```

---

## Dataset

### Trajectory Types

| ID prefix | Trajectory | Speed variants | Repetitions |
|-----------|-----------|----------------|-------------|
| A1b | Multi-sine excitation | — | 2 |
| B2 | Circle | slow, medium, fast | 3 each |
| B3 | Figure-eight | slow, medium, fast | 3 each |
| B5 | Helix | slow, medium, fast, Mellinger | 1–3 |
| B6 | Linear ramp | — | 4 |
| B7 | Oval | slow, medium, fast | 2–3 each |
| B8 | Star | slow, medium, fast | 2–3 each |
| B9 | Trefoil knot | slow, medium, fast | 5–18 each |
| B10 | Lissajous | slow, fast | 1 each |
| B11 | Random waypoints | — | 2 |
| B12 | Staircase climb | — | 1 |
| C4 | Battery drain (hover) | — | 2 |

**Total: 170 recordings, ~97.5 minutes of flight, ~603,942 timesteps at 100 Hz.**

> **Dataset availability note.** This repository includes the subset of recordings used directly by the three benchmark tasks. The complete NanoBench dataset — comprising all trajectory types, repetitions, and supplementary calibration flights — will be deposited on a public data repository upon acceptance of the accompanying paper. Instructions for accessing the full release will be provided here at that time.

### CSV Column Reference

Every file in `datasets/dataset/` shares the following 51-column schema:

| Column(s) | Units | Source | Description |
|-----------|-------|--------|-------------|
| `t` | s | Vicon host clock | Timestamp |
| `px`, `py`, `pz` | m | Vicon | Ground-truth position (world frame) |
| `qx`, `qy`, `qz`, `qw` | — | Vicon | Ground-truth orientation (unit quaternion, scalar-last) |
| `roll`, `pitch`, `yaw` | deg | Vicon | Ground-truth Euler angles (ZYX) |
| `vx`, `vy`, `vz` | m/s | Vicon | Ground-truth linear velocity (Savitzky-Golay differentiated) |
| `wx_vicon`, `wy_vicon`, `wz_vicon` | rad/s | Vicon | Ground-truth angular velocity (world frame) |
| `imu_acc_x`, `imu_acc_y`, `imu_acc_z` | g | Onboard IMU | Specific force (body frame) |
| `imu_gyro_x`, `imu_gyro_y`, `imu_gyro_z` | rad/s | Onboard IMU | Angular velocity (body frame) |
| `motor_motor_m1` … `motor_motor_m4` | PWM [0–65535] | Onboard | Per-motor PWM commands |
| `est_stateEstimate_x/y/z` | m | Onboard EKF | EKF position estimate |
| `est_stateEstimate_vx/vy/vz` | m/s | Onboard EKF | EKF velocity estimate |
| `est_stateEstimate_ax/ay/az` | m/s² | Onboard EKF | EKF acceleration estimate |
| `att_stateEstimate_roll/pitch/yaw` | deg | Onboard EKF | EKF attitude (Euler) |
| `att_stateEstimate_qx/qy/qz/qw` | — | Onboard EKF | EKF attitude (quaternion) |
| `sp_ctrltarget_x/y/z/yaw` | m, deg | Flight controller | Position and yaw setpoints |
| `pid_controller_roll/pitch/yaw` | — | Flight controller | PID roll/pitch/yaw outputs |
| `pid_controller_cmd_thrust` | — | Flight controller | PID thrust command |
| `pwr_pm_vbat` | V | Onboard power monitor | Battery voltage (10 Hz, forward-filled) |

### Metadata Files

Each CSV is paired with a `<filename>_metadata.yaml` containing trajectory type, speed category, repetition index, and nominal flight parameters.

---

## Environment Setup

A single conda environment covers all three tasks. Python 3.10 is recommended; PyTorch is needed for Tasks 1 and 2.

```bash
conda create -n nanobench python=3.10
conda activate nanobench

# Core scientific stack
pip install numpy scipy pandas matplotlib scikit-learn pyyaml tqdm joblib

# Task 1 and Task 2: deep learning
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# (CPU-only alternative: pip install torch torchvision)

# Task 2: imitation learning library
pip install imitation

# Task 2: MPPI
pip install pytorch-mppi

# Task 1: pytorch3d (for SO(3) rotation representation)
# Install from source if pip wheel is unavailable for your CUDA version:
pip install pytorch3d
# or: pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

All benchmark scripts are run from the repository root:

```bash
cd /path/to/nanobench
```

---

## Task 1 — System Identification

### Scientific Question

Can a data-driven dynamics model trained on motor PWM commands and onboard state estimates accurately predict next-step quadrotor states? How do physics-based, residual, and recurrent architectures compare on unseen trajectories?

### Baselines

| Model | Description |
|-------|-------------|
| **Naive** | Predict zero acceleration (constant-velocity) |
| **Physics** | Rigid-body model with fixed thrust and drag coefficients |
| **Residual** | Physics model + MLP residual correction |
| **Phys+Res** | Combined physics and learned residual in a single network |
| **LSTM** | Sequence-to-sequence recurrent model, no physics prior |

### Input / Output

- **Input:** Current state (12D: position, velocity, SO(3) log-map rotation, angular velocity) + motor speeds in rad/s (4D)
- **Output:** Next-step state (12D) over a configurable prediction horizon
- **Data source:** `datasets/dataset/` — flat CSVs are automatically converted to IDSIA format during the setup step

### How to Run

```bash
cd benchmarks/task1_sysid

# Step 1: Convert NanoBench CSVs to IDSIA format and split into train/test
python run_nanobench_sysid.py setup

# Step 2: Train all learned models (default 500 epochs)
python run_nanobench_sysid.py train

# Optional: specify device and epoch count
python run_nanobench_sysid.py train --epochs 300 --device cuda:0

# Step 3: Run inference on the test split
python run_nanobench_sysid.py test

# Step 4: Compute metrics and print LaTeX table
python run_nanobench_sysid.py compare

# Or run all steps in sequence
python run_nanobench_sysid.py all --epochs 500 --device cuda:0
```

### Outputs

```
benchmarks/task1_sysid/
├── data/
│   ├── train/       # IDSIA-format CSVs for training (~70% of trajectories)
│   └── test/        # IDSIA-format CSVs for evaluation (~30%)
├── checkpoints/     # Saved model weights (.pt)
└── results/         # Prediction CSVs, metric tables, comparison plots
```

---

## Task 2 — Control

### Scientific Question

Can a controller trained offline via imitation learning from logged Crazyflie PID data reproduce competitive motor-command sequences? How do a feedforward MLP, a recurrent LSTM, and an MPPI planner with a learned dynamics prior compare under closed-loop rollout evaluation?

### Baselines

| Baseline | Architecture | Description |
|----------|-------------|-------------|
| **BC-MLP** | Feedforward MLP | Behavioural cloning: state → motor PWM |
| **BC-LSTM** | Recurrent LSTM | Sequence-to-sequence behavioural cloning |
| **MPPI** | Sampling-based MPC | Model Predictive Path Integral with a learned forward dynamics model and BC prior as warm-start |

### Observation and Action Space

- **Observation (14D):** position (3), EKF velocity (3), EKF attitude quaternion (4), position setpoint (3), battery voltage (1)
- **Action (4D):** normalised motor PWM ∈ [0, 1] (divide by 65535 to recover raw values)

### How to Run

```bash
cd /path/to/nanobench   # repository root

# Train all three baselines and run evaluation
python benchmarks/task2_control/run_task2.py --mode all

# Train only
python benchmarks/task2_control/run_task2.py --mode train

# Evaluate a previously trained checkpoint
python benchmarks/task2_control/run_task2.py --mode eval

# Key options
python benchmarks/task2_control/run_task2.py \
    --mode all \
    --dataset-root datasets/dataset \
    --out-dir benchmarks/task2_control/results \
    --epochs 200 \
    --device cuda:0
```

### Outputs

```
benchmarks/task2_control/results/
├── bc_mlp_checkpoint.pt
├── bc_lstm_checkpoint.pt
├── dynamics_mlp_checkpoint.pt
├── split_metadata.json         # Train/val/test file assignment (reproducible)
├── scaler.pkl                  # StandardScaler fit on training observations
├── rollout_results.csv         # Per-trajectory rollout metrics
└── tables/
    └── task2_performance_table.tex / .csv
```

---

## Task 3 — State Estimation

### Scientific Question

How accurately does the Crazyflie's onboard EKF estimate position and attitude during aggressive trefoil-knot trajectories? Does estimation accuracy degrade systematically with battery discharge, and does it differ between the Mellinger and PID flight controllers?

### Evaluation

Onboard EKF estimates (`est_stateEstimate_*`, `att_stateEstimate_*`) are compared against Vicon ground truth using standard TUM/EuRoC trajectory evaluation metrics:

| Metric | Description |
|--------|-------------|
| **ATE RMSE / Mean** | Absolute Trajectory Error after Umeyama alignment |
| **RTE (1 m window)** | Relative Trajectory Error over 1 m path-length segments |
| **Velocity RMSE** | EKF vs. Vicon velocity, integrated over the flight |
| **Attitude RMSE** | Per-axis angle error (roll, pitch, yaw), pooled RMS |

The dataset contains 28 trefoil flights split across two controllers:

| Controller | Slow | Medium | Fast | Total |
|-----------|------|--------|------|-------|
| Mellinger | 5 | 5 | 5 | 15 |
| PID | 6 | 5 | 2 | 13 |

### How to Run

```bash
# Trefoil-specific evaluation (Mellinger vs PID, three speed regimes)
python benchmarks/task3_stateEst/run_task3_trefoil.py

```

### Outputs

```
benchmarks/task3_stateEst/results/
├── task3_trefoil_ate_comparison.pdf          # Bar chart: EKF ATE per speed
├── task3_trefoil_error_timeseries.pdf        # Position error with std bands (stacked)
├── task3_trefoil_error_2x2_with_overlay.pdf  # 2×2: error panels + XY trajectory overlay
├── task3_trefoil_trajectory_overlay.pdf      # 3D Vicon vs EKF, one panel per speed
├── task3_trefoil_velocity_error.pdf          # Velocity error per speed regime
├── task3_trefoil_per_axis_error.pdf          # X/Y/Z error components over normalized time
└── tables/
    ├── task3_trefoil_table.tex
    └── task3_trefoil_table.csv
```

---


## License

This dataset and codebase are released under the BSD 3-Clause License. See `LICENSE` for details.
