# PushBox data-scaling sweep

## What we are trying to do

**Question.** How does a world model's *real-world closed-loop* control
performance on PushBox scale with the amount of teleoperated data we collect,
and does that scaling depend on the model architecture?

**Why it matters — the decision this drives.** Collecting PushBox data on the
Trossen arm is the expensive, rate-limiting step. Before spending more operator
hours, we want evidence about where the return on additional data goes. Three
shapes of result lead to three different decisions:

- *Still climbing at full data* → collect more; the model is data-limited.
- *Plateaued by half data* → stop collecting; spend effort on the model,
  the reward, or the evaluation instead.
- *Architecture-dependent* → the "how much data" answer is not universal, and
  we should pick the architecture that reaches useful control soonest.

An earlier one-step *feature-loss* curve (see `docs/dinowm_learning_curves.md`)
already showed diminishing returns for DINO-WM — the marginal gain roughly
halved per doubling of data. But one-step latent loss cannot tell us whether a
model actually *controls the box*. This sweep replaces that proxy with the thing
we care about: closed-loop behaviour on the real robot.

**What this experiment is.** Train every candidate world model on nested
1/4, 1/2, 3/4, and full subsets of the PushBox training split, holding optimizer
steps and the held-out validation episodes fixed, so the only variable is how
much data the model saw. That yields a grid of deployable checkpoints:

|            | 1/4 | 1/2 | 3/4 | full |
| ---------- | :-: | :-: | :-: | :--: |
| DINO-WM (frozen DINOv2-s + predictor) | ✓ | ✓ | ✓ | ✓ |
| LeWM-small (scratch ViT-s, 384-d) | ✓ | ✓ | ✓ | ✓ |
| LeWM (scratch ViT-tiny, 192-d) | ✓ | ✓ | ✓ | ✓ |

Each cell also gets a paired pixel decoder so its predictions can be rendered
and inspected.

**Scope — where this repo stops.** There is no PushBox simulator:
`stable_worldmodel` registers no PushBox environment, and the dataset is a
recording, so nothing offline can execute a planned action and return the true
next observation. Closed-loop evaluation therefore happens on the physical arm
and is done separately by the operator. The deliverable here is the grid of
checkpoints **plus the exact normalization statistics needed to run each one**
(recorded per checkpoint in `split_manifest.json`) — feeding raw metres to a
model trained on z-scored actions is the quietest way to get a wrong rollout.

**How we will read the result.** Plot closed-loop success (or task metric)
against data fraction, one line per architecture. A rising line argues for more
collection; a flat line argues against. Because the three architectures use
different latent targets, they can only be ranked against each other by the
real-robot result — which is the entire point of measuring closed-loop rather
than validation loss. Before trusting any checkpoint on the arm, first confirm
it is action-sensitive (correct- vs shuffled-action, and 1/5/10/25-step recorded
rollouts) per Task 6 of
`docs/plans/2026-07-16-interactive-pushbox-world-simulator.md`.

## Controlled setup

Every cell of the sweep shares:

- the same deterministic episode-level split (`seed=3072`, `train_split=0.9`),
  leaving the same held-out 23 episodes (0.414 h, 7,382 clips) for validation;
- the same subset seed (`8317`), so fractions are nested — the 1/4 episodes are
  a subset of 1/2, which is a subset of 3/4 — and so a given fraction contains
  *identical episodes across architectures*;
- episode-atomic subsets, so no episode is split across train and validation;
- train-subset-only action and proprio normalization, recomputed per cell;
- 224 x 224 images, history 3, horizon 1, frameskip 1.

Because both trainers build clips with `num_steps=4` and `frameskip=1`, the
clip indexing is identical and the shared split is exact rather than
approximate.

| Fraction | Train data | Episodes | Clips | Share of clips |
| --- | ---: | ---: | ---: | ---: |
| 0.25 | 0.930 h | 50 | 16,587 | 25.0% |
| 0.5 | 1.856 h | 100 | 33,106 | 49.8% |
| 0.75 | 2.796 h | 155 | 49,858 | 75.0% |
| 1.0 | 3.727 h | 204 | 66,468 | 100.0% |

Fraction 1.0 selects exactly the episode set the earlier nominal "4 h" curve
used, so `dinowm_dinov2s_prop_4h` is a valid stand-in for the DINO-WM full-data
cell if you would rather not spend the GPU hour retraining it.

## Equal steps, not equal epochs

A fixed epoch count confounds the two things this sweep is trying to separate:
the small budgets would see less data *and* take proportionally fewer optimizer
steps, so they would be undertrained rather than data-limited. The sweep
therefore defaults to `MATCH_STEPS=1`, which scales the epoch count by roughly
`1/fraction` to hold optimizer steps constant against the full split.

| Fraction | DINO-WM epochs | LeWM epochs | Steps (approx.) |
| --- | ---: | ---: | ---: |
| 0.25 | 41 | 604 | 21.2k / 77.9k |
| 0.5 | 21 | 302 | 21.4k / 77.9k |
| 0.75 | 14 | 201 | 21.7k / 77.7k |
| 1.0 | 10 | 150 | 20.8k / 77.9k |

LeWM's `LinearWarmupCosineAnnealingLR` spans `max_epochs`, so the schedule
stretches with the epoch count rather than finishing early.

The cost of matched steps is that a small subset takes many more passes over
the same data and may overfit. Checkpoints are written throughout, so select
per cell on `validate/pred_loss` (LeWM) or `validate/pixels_loss_epoch`
(DINO-WM) rather than assuming the last epoch is best. Pass `MATCH_STEPS=0` to
train a fixed number of epochs instead.

## Run

```bash
bash scripts/submit_pushbox_data_scaling.sh                  # dinowm + lewm_small, all 4 fractions
bash scripts/submit_pushbox_data_scaling.sh dinowm           # one model, all fractions
bash scripts/submit_pushbox_data_scaling.sh lewm_small 0.25  # a single cell
bash scripts/submit_pushbox_data_scaling.sh all              # adds lewm (ViT-tiny, 192-d)
MATCH_STEPS=0 bash scripts/submit_pushbox_data_scaling.sh    # fixed epochs instead
```

Models: `dinowm` (frozen DINOv2-small + depth-6 causal predictor),
`lewm_small` (scratch ViT-small, 384-d latent), `lewm` (scratch ViT-tiny,
192-d latent). Checkpoints land in:

```text
$STABLEWM_HOME/checkpoints/pushbox/dinowm_dinov2s_prop_{25,50,75,100}pct/
$STABLEWM_HOME/checkpoints/pushbox/lewm_small_{25,50,75,100}pct/
$STABLEWM_HOME/checkpoints/pushbox/lewm_{25,50,75,100}pct/
```

Each directory holds `weights_epoch_*.pt`, `training_config.yaml`, and
`split_manifest.json`. The manifest is the deployment contract: it records the
exact train episode indices, validation episode indices, clip and frame counts,
epoch and step budget, and the action/proprio mean and std the model expects.
Feeding raw metres to a model trained on z-scores is the most likely way to get
a silently wrong rollout on the robot, so read the statistics from the manifest
rather than recomputing them.

Expected wall time per cell on one H200: roughly 1.5 h for DINO-WM and 3.5-5 h
for LeWM, so the default 8-cell sweep is about 25 GPU-hours. The LeWM script
uses a job-id independent run directory, so a requeued job resumes from its last
checkpoint instead of restarting.

## Reading the result

Only compare like with like. `validate/pixels_loss_epoch` is comparable across
DINO-WM cells because its target is the frozen backbone, which is identical
everywhere. It is **not** comparable to LeWM's `validate/pred_loss`, whose
target is a jointly trained latent with a per-run arbitrary scale — the two
architectures can only be ranked against each other by the real-robot
closed-loop result, which is the point of the sweep.

One-step validation loss also cannot tell you whether a model responds to
actions at all. Before trusting any cell on the arm, run the correct-action
versus shuffled-action comparison and the 1/5/10/25-step recorded-trajectory
rollouts described in `docs/plans/2026-07-16-interactive-pushbox-world-simulator.md`
(Task 6).
