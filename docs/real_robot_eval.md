# Real-Robot PushBox Evaluation

## Implementation update — July 20, 2026

This document summarizes the real-robot evaluation pipeline introduced in
commit `8065dde` (`Add real robot PushBox evaluation pipeline`). The pipeline
supports safe interactive PushBox evaluation, goal-conditioned latent-space
planning, reproducible run records, and offline model diagnostics.

### Runtime architecture

- `real_robot_eval.py` owns the camera, UI, Trossen arm connection, safety
  checks, command execution, and run recording. It runs with the ROS-compatible
  Python 3.12 environment from the sibling `wm_data_collection` repository.
- `real_robot_planner.py` loads LeWM and performs CEM planning in this
  repository's Python 3.10 virtual environment.
- The two processes communicate through a local authenticated Unix socket. No
  ROS or Trossen modules are imported into the model process.
- `scripts/eval_pushbox_real.sh` supplies the commissioned robot pose,
  workspace, planner defaults, and explicit live/dry-run modes.

### Interactive evaluation

The evaluator now provides:

- a blocking home/startup sequence followed by a fixed downward-facing
  end-effector pose;
- manual XY control using the same arrow directions and 2.5, 5, and 10 mm
  action magnitudes used during data collection;
- goal capture, plan preview, autonomous start/pause, reset, success/failure
  labels, and controlled shutdown;
- optional startup goals loaded from a 224x224 image or an explicitly indexed
  frame in a dataset MP4;
- one-step receding-horizon MPC: CEM plans a full horizon, but only its first
  action is executed before observing and replanning;
- finite workspace, pose, orientation, action-norm, planner-age, camera-age,
  and motion-completion checks;
- post-command observations captured only after the arm reaches its target and
  a newer camera frame arrives;
- latched handling for genuine unsafe states, while normal paused/manual pose
  checks do not interpret instantaneous Cartesian velocity as a fault.

The configured workspace and pose are:

| Setting | Value |
| --- | --- |
| Start XY | `(0.14, 0.0185)` m |
| Safe/fixed Z | `0.15 / 0.03` m |
| X bounds | `[0.0183, 0.45]` m |
| Y bounds | `[-0.26, 0.26]` m |
| End-effector orientation | `[0, pi/2, 0]` |
| Settled-speed observation gate | `0.1 m/s` |
| Maximum collected action norm | `0.010 m` per step |

The settled-speed value is an observation gate, not the commanded robot speed.
The controller also waits for the nominal trajectory duration and verifies the
target pose before accepting a post-motion image.

### CEM planner

The real planner:

- loads `pushbox/lewm/weights_epoch_146.pt` by default;
- reproduces the train-only action mean and sample standard deviation;
- maintains the model's three-frame latent/action history across MPC steps;
- recursively predicts a complete latent trajectory for every candidate plan;
- minimizes terminal projected-latent MSE to the encoded goal, with optional
  action and smoothness regularization;
- warm-starts from the preceding plan and keeps the returned plan and reported
  cost aligned;
- validates all planner outputs before they cross the robot boundary.

The default `keyboard` action mode prevents CEM from exploiting arbitrary
continuous actions absent from keyboard collection. It searches zero plus the
eight normalized arrow-key directions at each enabled magnitude. With
`--action-cap 0.005`, this produces 17 exact actions: zero and eight directions
at both 2.5 and 5 mm. `--action-mode continuous` remains available for an
explicit comparison.

### Artifact contract and run records

`config/real_robot_eval.json` records the expected checkpoint, dataset,
camera-transform hashes, train/validation split, action normalization, latent
dimension, history length, action axes, and control rate. Live evaluation fails
closed on a contract mismatch unless the commissioning-only override is given.

Every invocation creates a timestamped directory under `real_robot_runs/` with:

- resolved arguments and artifact metadata in `metadata.json`;
- flushed state transitions, commands, timing, faults, and outcomes in
  `events.jsonl`;
- lossless 224x224 RGB goal and observation PNG files;
- requested, accepted, executed, and planner-feedback actions;
- complete selected CEM plans, measured poses, latent goal distance, and solver
  latency.
- projected observation latents in `latents/z/` and accepted-action-conditioned
  one-step predictions in `latents/z_hat/` for every executed autonomous step.

`real_robot_runs/` and derived `outputs/` remain ignored by Git so hardware data
and large diagnostic artifacts are not accidentally committed.

### Offline diagnostics

| Tool | Purpose |
| --- | --- |
| `scripts/replay_real_cem_latents.py` | Re-encode recorded PNGs, reconstruct planner history, replay each selected CEM plan, decode every predicted horizon state, and report predicted-versus-real one-step errors. |
| `scripts/render_real_latent_alignment.py` | Decode saved live `z`/`z_hat` arrays and create the three-row time-aligned raw/encoded/predicted figure. |
| `scripts/probe_pushbox_latent_rollout.py` | Apply a constant synthetic XY action from one real image and decode the open-loop latent trajectory. |
| `scripts/record_real_constant_xy.py` | Safely record a real constant-XY robot trajectory with aligned pre/post-action images and measured states. |
| `scripts/analyze_real_latent_rollout.py` | Compare encoded real frames with the corresponding predicted latent rollout and generate plots, tables, and video. |
| `scripts/run_real_plus_x_latent_probe.sh` | Run the constant-`+X` recorder and analyzer as one preflight-safe workflow. |

Decoder-based visualizations require a projected-latent decoder trained for the
same world-model checkpoint. The CEM replay tool rejects mismatched decoder and
world-model checkpoints by default. Only the first predicted state of a plan is
directly comparable with the next real image because live MPC replans after
every action.

### Commands

Validate configuration and artifacts without connecting to the arm:

```bash
scripts/eval_pushbox_real.sh --dry-run --preflight
```

Open the camera and planner without robot motion:

```bash
scripts/eval_pushbox_real.sh --dry-run
```

Run the configured live evaluator:

```bash
scripts/eval_pushbox_real.sh
```

Example constrained horizon-10 evaluation:

```bash
scripts/eval_pushbox_real.sh \
  --horizon 10 \
  --action-cap 0.005 \
  --action-mode keyboard \
  --num-samples 1024 \
  --iterations 10 \
  --elite-count 32 \
  --max-actions 100
```

Interactively select a goal from a dataset episode before the original runtime
pipeline starts:

```bash
scripts/eval_pushbox_real.sh \
  --goal-video datasets_videos/20260714_145344/ep_000.mp4
```

The selector uses Left/Right for navigation. Press `1` or `5` to set its skip
mode and Enter to confirm the displayed frame. `Esc`, `q`, or window close
cancels without starting the planner, camera, or robot. Add
`--goal-video-frame 42` to bypass the selector; this explicit form is required
when combining a video goal with `--preflight`.

Replay selected plans from a recorded run:

```bash
.venv/bin/python scripts/replay_real_cem_latents.py \
  --run real_robot_runs/RUN_ID \
  --plan-step 1 --plan-step 10 --plan-step 20
```

Render the saved live one-step predictions beneath their target timestamps:

```bash
scripts/run_real_plus_x_latent_probe.sh real_robot_runs/RUN_ID
```

The wrapper detects normal evaluation runs with saved `z`/`z_hat` artifacts and
uses the three-row alignment renderer. Options after the run directory are
forwarded to it, for example `--trial-id 2` or `--device cpu`. It still detects
and analyzes its original constant-XY recorder format as before.

### Verification status

As of July 20, 2026:

- shell and Python syntax checks pass;
- the epoch-146 planner loads against the artifact manifest;
- constrained CEM outputs were verified to contain only exact keyboard-action
  vocabulary entries;
- all 25 real-evaluation, latent-alignment, and constant-trajectory unit tests
  pass.
