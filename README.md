
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints &amp; Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

### PushBox training pipeline

The merged PushBox dataset lives at
`$STABLEWM_HOME/datasets/pushbox_pilot_train.h5`. It contains 227 episodes,
74,531 transitions, and 73,850 valid four-frame clips at the original 5 Hz
collection rate. The dedicated configuration uses `frameskip: 1` and loads the
two columns consumed by LeWM (`pixels` and `action`):

```bash
python train.py --config-name=pushbox
```

The split keeps episodes atomic while balancing the validation set by usable
clip count. By default, training uses batch size 128 for 150 epochs and logs
training and validation losses to Weights & Biases. Point it at a different
compatible HDF5 file without editing the config:

```bash
PUSHBOX_DATASET=/absolute/path/to/pushbox.h5 \
  python train.py --config-name=pushbox
```

Portable model weights and their `config.json` are written to
`$STABLEWM_HOME/checkpoints/pushbox/lewm/weights_epoch_<N>.pt`. To submit the
world-model job and a decoder job that starts only after successful training:

```bash
bash scripts/submit_pushbox_pipeline.sh
```

Stable-pretraining's full-state resume checkpoints and CSV metrics are stored
in the run-cache directory printed near the start of each training job.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set `policy` either to a public
Hugging Face model repo or to a local `.pt` checkpoint path relative to
`$STABLEWM_HOME/checkpoints`:

```bash
# Public pretrained PushT LeWM checkpoint
python eval.py --config-name=pusht policy=quentinll/lewm-pusht

# Local checkpoint
python eval.py --config-name=pusht policy=lewm/weights_epoch_10.pt
```

### Real PushBox planning

The real-robot evaluator reuses the camera transform, Cartesian startup pose,
fixed height, arrow controls, and shutdown sequence from the sibling
`wm_data_collection` repository. The controller runs with ROS-compatible Python
3.12 while LeWM planning runs in this repository's virtual environment over a
local authenticated Unix socket.

See [the dated implementation summary](docs/real_robot_eval.md) for the runtime
architecture, safety behavior, planner changes, run-record format, and offline
diagnostic tools added on July 20, 2026.

The configured launcher defaults to **live robot execution** with the
commissioned start pose and X/Y bounds in `scripts/eval_pushbox_real.sh`.
Validate it first, then use its explicit dry-run mode to open the camera and
planner without connecting to or commanding the arm:

```bash
scripts/eval_pushbox_real.sh --dry-run --preflight
scripts/eval_pushbox_real.sh --dry-run
```

Instead of capturing a goal with `g`, open a recorded dataset video in the goal
frame selector. This happens before the planner, camera, or robot starts:

```bash
scripts/eval_pushbox_real.sh \
  --goal-video datasets_videos/20260714_145344/ep_000.mp4
```

In the selector, use Left/Right to move, press `1` or `5` to choose the number
of frames skipped per key press, and press Enter to confirm. `Esc`, `q`, or
closing the window cancels before hardware access. For a scripted run, bypass
the UI with `--goal-video-frame 42`; preflight runs require this explicit frame
index because `--preflight` never opens a display.

An already-extracted 224x224 RGB frame can be supplied with `--goal-image`.
Relative goal paths are resolved from this repository even though the hardware
process launches from the sibling collector repository.

The default model is `pushbox/lewm/weights_epoch_146.pt`, selected by validation
prediction loss. Controls are:

- Arrow keys: manual XY motion while autonomous execution is paused.
- `1`, `2`, `3`: select the same 2.5, 5, or 10 mm manual step as collection.
- `g`: capture the current transformed camera frame as a new trial goal.
- `v`: compute a preview plan from the current scene without moving the arm.
- `p`: start or pause autonomous planning and execution.
- `SPACE`: cancel autonomous/manual motion by holding the measured pose.
- `r`: return the arm to the configured start XY while paused.
- `s` / `f`: label the active trial as success or failure.
- `q` or window close: pause, then run the collector's home-to-zero shutdown;
  after a latched fault, automatic recovery motion is deliberately skipped.

A safe goal workflow is: manually arrange the desired scene, press `g`, reset
the box and arm to the evaluation start, press `v` to inspect the proposed plan,
then press `p`. Focus loss, stale/over-age planning input, planner failure,
measured pose deviation, a blocked workspace action, or the maximum per-trial
action count pauses execution. A paused trial retains its action count; capture
a new goal to start a new budget.

Every interactive invocation writes a timestamped directory under
`real_robot_runs/` containing resolved metadata, flushed JSONL events, goal and
planning frames as lossless RGB PNG images, complete plans, timing,
requested/accepted/executed actions, measured poses, and the final outcome. For
every executed autonomous action it also saves projected encoder `z[t]` under
`latents/z/` and the one-step predictor result `z_hat[t+1]` under
`latents/z_hat/`, conditioned on the accepted physical XY action.
Autonomous control is one-step receding-horizon MPC: only the first action of
each CEM plan is sent. Before encoding the next observation, the evaluator
waits for the arm to reach its target and fall below the configured settled
speed, then requires a camera frame received strictly after that verification.
By default, `--action-mode keyboard` restricts every CEM candidate to the exact
zero/8-way actions used by the collector at the enabled 2.5, 5, and 10 mm
magnitudes. The selected `--action-cap` removes larger magnitudes; for example,
`--action-cap 0.005` searches 17 actions (zero plus eight directions at 2.5 and
5 mm). Use `--action-mode continuous` only for an explicit A/B comparison.
The default checkpoint, dataset, normalization, camera parameters, and image
transform are checked against
`config/real_robot_eval.json`; `--allow-artifact-mismatch` is an explicit
commissioning-only override.

Recorded trials can be replayed without the robot. This re-encodes each saved
PNG, reconstructs the exact three-state/action planner context, rolls out the
recorded selected CEM plan, and decodes one predicted image per horizon step:

```bash
.venv/bin/python scripts/replay_real_cem_latents.py \
  --run real_robot_runs/20260717_202906_1203310 \
  --plan-step 1 --plan-step 10 --plan-step 20 --plan-step 30
```

The generated `outputs/real_cem_latent_replay_<run>/index.html` links contact
sheets, raw latent arrays, per-step metrics, and an actual goal-distance plot.
For horizon 10, each selected plan contains `predicted_001.png` through
`predicted_010.png`. Only prediction 1 is directly comparable with the next
real observation because the live controller replans after every action. The
tool refuses a decoder trained against a different world-model checkpoint by
default.

New runs with live latent artifacts can be rendered directly as a time-aligned
three-row figure:

```bash
.venv/bin/python scripts/render_real_latent_alignment.py \
  --run real_robot_runs/RUN_ID \
  --trial-id 1
```

The rows are raw `observation[t]`, `decoder(z[t])`, and
`decoder(z_hat[t])`. The first cell in the prediction row is empty; every later
prediction is shifted under the real next observation it predicts. The final
saved prediction has no later real frame to align with and remains available in
the generated `decoded_z_hat/` directory.

After checking the physical workspace and keeping the emergency stop ready,
start the configured live launcher with:

```bash
scripts/eval_pushbox_real.sh
```

The launcher currently uses start XY `(0.14, 0.0185)` m, safe/fixed Z
`0.15/0.03` m, X bounds `[0.0183, 0.45]` m, and Y bounds `[-0.26, 0.26]` m.
Its `0.1 m/s` settled-speed value is a post-command observation gate, not a
commanded-velocity limit; ordinary paused and manual pose checks do not treat
instantaneous Cartesian velocity as a latched fault.
Command-line values appended to the launcher override these configured values.
Run dry mode first and confirm planner latency is comfortably below the intended
control cadence before enabling motion.

## Pretrained Checkpoints

Pretrained LeWM checkpoints for each environment are mirrored on the Hugging Face
Hub (model repos), alongside the datasets (dataset repos) in the same collection:

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop)
is available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e):

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

### From the Drive archive

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Place the extracted files under `$STABLEWM_HOME/` and load via:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

`AutoCostModel` accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

### From the Hugging Face mirror

The HF model repos ship the LeWM checkpoint as a `weights.pt` (state dict) plus a
`config.json` describing the model. Convert once to produce the `_object.ckpt`
that `eval.py` expects:

```bash
# download weights.pt + config.json
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

# convert to object checkpoint under $STABLEWM_HOME/pusht/lewm_object.ckpt
python - <<'PY'
import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**cfg["predictor"]),
    action_encoder=Embedder(**cfg["action_encoder"]),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
PY
```

After conversion, load via `swm.policy.AutoCostModel('pusht/lewm')` as usual.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
