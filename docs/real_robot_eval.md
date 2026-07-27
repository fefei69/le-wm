# Real-Robot PushBox Evaluation

## Implementation update — July 23, 2026

This document summarizes the real-robot evaluation pipeline introduced in
commit `8065dde` (`Add real robot PushBox evaluation pipeline`). The pipeline
supports safe interactive PushBox evaluation, goal-conditioned latent-space
planning, reproducible run records, and offline model diagnostics.

### Runtime architecture

- `real_robot_eval.py` owns the camera, UI, Trossen arm connection, safety
  checks, command execution, and run recording. It runs with the ROS-compatible
  Python 3.12 environment from the sibling `wm_data_collection` repository.
- `real_robot_planner.py` loads either LeWM or DINO-WM and performs CEM
  planning in this repository's Python 3.10 virtual environment.
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
- aligned HDF5 episode replay with operator-selected initial/goal frames and
  automatic restoration of the initial saved end-effector XY;
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

### CEM planners

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

The default `--solver cem` retains Gaussian CEM: it samples continuous XY
vectors and then constrains them to the keyboard vocabulary. The additional
`--solver categorical-cem` maintains one probability vector over the exact
keyboard tokens for each horizon step and samples token indices directly:

```bash
scripts/eval_pushbox_real.sh --solver categorical-cem
```

Both variants use the same horizon, sample/iteration/elite counts, latent
rollout, objective, and one-action receding-horizon execution. Categorical CEM
updates its distribution from elite token frequencies with
`--categorical-alpha` (default `0.7`) and preserves exploration with
`--categorical-min-prob` (default `0.01`). It requires
`--action-mode keyboard`. The next MPC call warm-starts by shifting the final
probability vectors left and resetting the last horizon step to uniform.

Categorical preview and autonomous-step records include
`action_probabilities` with shape `(horizon, vocabulary_size)` and the matching
`action_vocabulary`, so convergence and competing directional modes can be
inspected directly in `events.jsonl`.

### DINO-WM backend

`--world-model dinowm` selects the DINO-WM pipeline trained on the `hpc`
branch. It preserves the existing UI, goal-image/video/dataset selection,
Gaussian or categorical CEM, one-action MPC execution, safety gates, and run
directory layout.

The runtime contract is:

- `weights_epoch_10.pt` and its sibling `config.json` provide the full frozen
  DINOv2-small backbone, action/proprio encoders, and causal predictor;
- the sibling `split_manifest.json` provides the exact episode-subset action
  and proprio normalization used for that checkpoint;
- the evaluator measures proprioception as `[x, y, vx, vy]`, exactly matching
  data collection, after the arm has reached the commanded pose;
- the predictor context contains 256 spatial DINO patches per frame. Every
  observed context state receives the action actually associated with that
  state, and every recursive prediction receives the candidate action for that
  rollout step;
- goal distance and CEM terminal cost compare only the 384-dimensional image
  portion of each predicted patch against the goal image. Proprioception and
  action condition the dynamics but do not become part of the visual goal;
- saved `z` and `z_hat` remain one-dimensional `.npy` arrays for compatibility,
  with the 256 x 384 token structure recorded in `metadata.json` as
  `latent_semantic_shape`.

The loader reconstructs the known DINOv2-small architecture locally and
strictly loads its weights from the world-model checkpoint. A Hugging Face
download is therefore not required for evaluation.

Because spatial DINO rollouts are much larger than LeWM's single 192-D vector,
`--cem-batch-size` bounds candidate memory. When selected through the shell
launcher, DINO-WM defaults to categorical CEM with horizon 5, 64 samples,
3 iterations, 8 elites, and batches of 32. These are operational starting
values, not a claim that the optimizer is tuned; explicit command-line values
still override them.

Expected checkpoint layout:

```text
dinowm_dinov2s_prop_4h/
├── config.json
├── split_manifest.json
└── weights_epoch_10.pt
```

Run the no-hardware gate first:

```bash
scripts/eval_pushbox_real.sh --dry-run --preflight \
  --world-model dinowm
```

This selects
`stable-wm/checkpoints/dinowm_dinov2s_prop_4h/weights_epoch_10.pt` by default.
An explicit `--checkpoint` still overrides it.
Use `--model-manifest /other/path/split_manifest.json` only when the manifest
is not beside the checkpoint.

### Artifact contract and run records

`config/real_robot_eval.json` records the expected checkpoint, dataset,
camera-transform hashes, train/validation split, action normalization, latent
dimension, history length, action axes, and control rate. Live evaluation fails
closed on a contract mismatch unless the commissioning-only override is given.
For DINO-WM, its LeWM-specific checkpoint/latent entries are replaced by the
selected checkpoint's `split_manifest.json`; the shared dataset, camera,
transform, control-rate, action-axis, and workspace contracts still apply.

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
- the selected plan's complete compounded latent trajectory in
  `latents/z_hat_rollout/`, saved as one `(horizon, latent_dim)` float32 matrix
  for every executed autonomous step.
- for DINO-WM runs, the exact observation proprioception used by the model and
  the authoritative flattened/semantic latent shapes.

`real_robot_runs/` and derived `outputs/` remain ignored by Git so hardware data
and large diagnostic artifacts are not accidentally committed.

#### Full imagined trajectory

LeWM and DINO-WM both recursively roll the selected CEM action sequence through
their world model before returning the first MPC action. The evaluator now
retains those states instead of discarding them after planning:

```text
latents/z/trial_NNN_step_MMM.npy
    z[t], shape (latent_dim,)

latents/z_hat/trial_NNN_step_MMM.npy
    accepted-action one-step z_hat[t+1], shape (latent_dim,)

latents/z_hat_rollout/trial_NNN_step_MMM.npy
    selected-plan z_hat[t+1:t+H | t], shape (horizon, latent_dim)
```

Row `k-1` of `z_hat_rollout` is `z_hat[t+k | t]`. Each predicted state is fed
back to predict the next row, so rollout error compounds over the horizon. The
complete state path for visualization is:

```python
from pathlib import Path
import numpy as np

run = Path("real_robot_runs/RUN_ID")
name = "trial_001_step_001.npy"
z_t = np.load(run / "latents/z" / name)
z_hat = np.load(run / "latents/z_hat_rollout" / name)
z_t_through_h = np.concatenate([z_t[None], z_hat], axis=0)  # (H + 1, D)
```

The evaluator performs fresh receding-horizon planning for every action, so
every `autonomous_step` has an `imagined_latent_rollout` path and records
`planner_replanned: true`. The rollout is aligned with the selected `plan`
stored in that event. The legacy one-step `z_hat` is deliberately different in
meaning: it is recomputed using the accepted robot action after deadband,
coordinate conversion, and workspace clipping. Its value should match rollout
row 0 when the selected first action is accepted unchanged, but may differ when
the safety layer changes that action.

For DINO-WM, every spatial `(num_patches, pixel_dim)` state is flattened to the
same `latent_dim` convention used by the existing `z` and `z_hat` files. Recover
the spatial shape using `metadata.json`'s
`model_runtime.latent_semantic_shape`.

### Named real-robot experiments

`config/real_robot_experiments.json` is a small registry for repeatable
case/method comparisons. A case fixes the source video and both frame indices;
a method fixes the repository, world model, solver, and planner overrides:

```json
{
  "cases": {
    "test01": {
      "dataset_episode": "datasets_videos/20260715_180541/ep_005.mp4",
      "initial_step": 80,
      "goal_step": 160
    }
  }
}
```

The checked-in methods cover:

- `lewm_cem`
- `dinowm_categorical_cem`
- `discrete_cem`
- `discrete_mcts`

List the registry or verify one combination without hardware:

```bash
.venv/bin/python scripts/run_real_robot_experiment.py --list

.venv/bin/python scripts/run_real_robot_experiment.py \
  --case test01 \
  --method discrete_mcts \
  --preflight
```

Run one live, operator-supervised experiment:

```bash
.venv/bin/python scripts/run_real_robot_experiment.py \
  --case test01 \
  --method discrete_mcts \
  --execute
```

Run multiple independent repetitions of one case/method with the wrapper:

```bash
scripts/run_real_robot_repetitions.sh \
  --case test01 \
  --method discrete_mcts \
  --repetitions 5 \
  --execute
```

The wrapper pauses before every live repetition so the operator can restore the
box and clear the workspace. Each repetition is a separate timestamped run with
`trial_001`; normal postprocessing finishes before the next prompt. `--trial-id`
is intentionally not used as a repetition counter. The defaults at the top of
the script can also be overridden with `CASE_NAME`, `METHOD_NAME`, and
`REPETITIONS` environment variables.

An explicit `--execute`, `--dry-run`, or `--preflight` is required so selecting
a config cannot accidentally move the arm. Real method matrices are
intentionally not run unattended: arrange the box, start autonomy, label/quit
the run, and then launch the next method.

After the evaluator exits successfully, the runner:

1. identifies the one new `real_robot_runs/RUN_ID` directory;
2. writes `RUN_ID/experiment.json` containing the exact case, method, command,
   repository, timestamps, and exit codes;
3. invokes `scripts/postprocess_discrete_real_run.sh` for the configured trial;
4. records the resulting `analysis/overview_trial_NNN` path in
   `experiment.json`.

Postprocessing is automatic by default. Use `--skip-postprocess` only when the
run is intentionally incomplete. `--fps 10` overrides output video FPS, and
extra evaluator flags can be appended after `--`, for example:

```bash
.venv/bin/python scripts/run_real_robot_experiment.py \
  --case test01 --method discrete_cem --execute -- \
  --max-actions 30
```

Relative method repository paths are resolved from the `le-wm` repository;
relative dataset-video paths are resolved inside the selected method's
repository. This lets the same named case launch either `le-wm` or the sibling
`discrete-la-wm` checkout.

### Offline diagnostics

| Tool | Purpose |
| --- | --- |
| `scripts/postprocess_real_run.py` | Create a side-by-side raw-observation/goal MP4, plot recorded planner cost, and add optional box/EE pixel-tracking checks for one trial. |
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

`render_real_latent_alignment.py` also accepts the DINO patch decoder produced
by `train_dino_decoder.py`. It reshapes the flattened live artifacts back to
their recorded `(256, 384)` token grid and rejects a decoder trained for a
different world-model checkpoint:

```bash
.venv/bin/python scripts/render_real_latent_alignment.py \
  --run real_robot_runs/RUN_ID \
  --decoder /path/to/decoder_vqvae/decoder_best.pt
```

The heavier `replay_real_cem_latents.py` tool still reconstructs LeWM planner
history internally and is therefore LeWM-only. DINO-WM runs should use their
live saved `z/z_hat` artifacts with `render_real_latent_alignment.py`.

The lightweight run overview does not load the model or require a decoder:

```bash
.venv/bin/python scripts/postprocess_real_run.py \
  --run RUN_ID \
  --trial-id 1
```

It writes `raw_vs_goal.mp4`, `planning_loss.png`,
`planning_metrics.csv`, and `summary.json` under
`RUN/analysis/overview_trial_001/`. Each raw video frame is the observation
used for that autonomous action. “Planning loss” means the recorded CEM
objective in `autonomous_step.cost`; it is not a world-model training loss.
The plot uses elapsed monotonic wall time, while the CSV also includes step,
latent goal distance, and solve time. `--run` accepts either a full/relative
run-directory path or a bare directory name found under `real_robot_runs/`.

As an additional check, it also writes `tracking_check.mp4`,
`tracking_errors.png`, and `tracking_metrics.csv`. The box pose is the centroid
and minimum-area rectangle fitted to the large red patch. The fitted four
corners and long-axis angle are recorded; goal-relative orientation error is
reported in degrees modulo 180. A square-symmetric marker is still ambiguous
modulo 90. The EE proxy is the small red pusher-tip centroid with temporal
continuity. Missing detections are recorded as NaN and do not affect the
primary video or planner-cost artifacts. Goal-relative box and EE distances
remain separate and use image pixels.

For new runs, `s`/`f`, automatic `at_goal`, and action-budget completion save a
terminal frame only after robot settling and a strictly newer camera receipt.
The raw/tracking videos append this `terminal_post_action` sample, and
`summary.json` reports explicit terminal box-center, box-orientation, and EE
errors plus terminal position progress. After pressing `s` or `f`, wait for
`saved settled terminal observation` before pressing `q`; quitting remains an
immediate safety action and can interrupt terminal capture. Legacy runs without
a terminal frame are explicitly marked and retain last-pre-action semantics.

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

### Replay an aligned dataset start and goal

Use `--dataset-episode ID_OR_VIDEO` when the initial scene should also come
from the planner's HDF5 dataset. It accepts either the merged integer episode
ID or its readable source-video path, for example
`datasets_videos/20260715_180541/ep_003.mp4`. The path form validates the
video inventory against the HDF5 `source_files_json` order and resolves this
example to merged episode 166. The original path and both IDs are retained in
run metadata.

The selector first asks for an initial step and then a later goal step.
Left/Right navigates, `1`/`5` changes the stride, Enter accepts the displayed
step, Backspace returns to initial-step selection, and Esc/Q cancels before the
planner, camera, or robot starts.

```bash
# Interactive selection with camera/UI but no robot motion.
scripts/eval_pushbox_real.sh --dry-run \
  --dataset-episode datasets_videos/20260715_180541/ep_003.mp4

# Reproducible no-display preflight.
scripts/eval_pushbox_real.sh --dry-run --preflight \
  --dataset-episode datasets_videos/20260715_180541/ep_003.mp4 \
  --initial-step 10 --goal-step 80

# Live replay. This launcher defaults to robot execution.
scripts/eval_pushbox_real.sh \
  --dataset-episode datasets_videos/20260715_180541/ep_003.mp4 \
  --initial-step 10 --goal-step 80

# The legacy merged integer ID remains supported.
scripts/eval_pushbox_real.sh --dry-run --dataset-episode 166
```

Both images come directly from the selected episode's `pixels` rows. The
initial end-effector XY comes from `state[initial_step, :2]`, replaces
`--start-x/--start-y`, and is checked against the commissioned workspace before
the robot is connected. Live startup follows the existing home -> safe-Z ->
fixed-Z path at that XY, and `r` resets to the same saved position.

The live window shows `CURRENT | DATASET START | GOAL`. Run records preserve
both images, episode and step IDs, dataset row indices, and the selected XY.
Only the arm position is restored automatically: box position/yaw are not
recorded in this dataset, so manually match the box to `DATASET START` before
pressing `v` or `p`. The goal step must be later than the initial step.

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
- all 42 real-evaluation, DINO runtime/decoder, latent-alignment, and
  constant-trajectory unit tests
  pass.
- dataset replay preflight loads aligned HDF5 start/goal frames and saved
  initial end-effector XY without opening the display, camera, or robot.
