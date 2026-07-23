# ViT-Small Predictor-Capacity Sweep

This note records the motivation, design, and interpretation criteria for the
PushBox predictor-depth sweep submitted on 2026-07-22. The experiment asks
whether the autoregressive predictor became a bottleneck when the visual
encoder was enlarged from ViT-tiny to ViT-small.

## Motivation

Moving from ViT-tiny to ViT-small changed the representation width from 192 to
384 and increased the encoder from 5.5M to 21.6M parameters. Parts of the
predictor scaled automatically with the latent width, increasing it from 10.8M
to 24.2M parameters, but its depth remained fixed at six layers. Consequently,
the predictor-to-encoder parameter ratio fell from about 2:1 to 1.1:1.

The completed ViT-small depth-6 run produced two apparently conflicting
results:

- Its projected-latent decoder was better than the matched ViT-tiny probe:
  validation MSE improved from 0.03113 to 0.02942 (-5.5%), and PSNR improved
  from 28.010 dB to 28.258 dB.
- Its held-out world-model prediction loss was worse: the final
  `validate/pred_loss_epoch` increased from 0.004600 to 0.004995 (+8.6%). The
  best value was also about 8% worse than the ViT-tiny best.

One explanation is that ViT-small retains more useful visual information but
the six-layer predictor cannot model the dynamics of the richer latent. Other
plausible explanations are that the richer representation contains
unpredictable frame detail, or that the larger system overfits. The sweep is
designed to distinguish these explanations.

## Hypotheses

### H1: the predictor is capacity-limited

Increasing predictor depth should lower both training and validation prediction
loss. The late-epoch validation improvement should be sustained rather than an
isolated noisy minimum, and the train/validation gap should remain stable or
shrink. Multi-step rollout accuracy should also improve.

### H2: additional capacity mainly increases overfitting

Training prediction loss should improve while validation prediction loss stays
flat or worsens. The train/validation gap will widen. If this occurs, additional
data, regularization, or a more predictability-focused target is more promising
than a still-larger predictor.

### H3: representation predictability is the bottleneck

Neither training nor validation prediction loss should materially improve with
depth. The better decoder would then indicate that ViT-small preserves more
image information without establishing that this information is useful or
predictable for dynamics.

## Controlled experiment

The completed depth-6 model is the baseline. Two models are being retrained
from scratch with only `model.predictor.depth` and the resulting parameter count
changed:

| Variant | Predictor depth | Predictor parameters | WM job | Decoder job |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 6 | 24.2M | `14518368` | `14518369` |
| Sweep | 8 | 32.3M | `14553765` | `14553766` |
| Sweep | 10 | 40.4M | `14553767` | `14553768` |

At submission, both new WM jobs were pending on `QOSGrpGRES`. Each decoder job
has an `afterok` dependency on its corresponding WM job.

The following are held fixed:

- ViT-small encoder, 384-dimensional projected latent, and patch size 14
- predictor settings other than layer count: 16 heads, head dimension 64, MLP
  width 2048, and dropout 0.1
- PushBox dataset and deterministic episode-grouped split
- seed 3072, history length 3, and one-step prediction target
- batch size 128, AdamW learning rate `5e-5`, and weight decay `1e-3`
- 150 epochs, bf16 mixed precision, and SIGReg weight 0.09

Checkpoints and logs are isolated by variant:

| Depth | Checkpoint directory | WM log pattern | Decoder log pattern |
| ---: | --- | --- | --- |
| 8 | `pushbox/lewm_small_pred_d8` | `outputs/pushbox-wm-small-pred-d8-*.out` | `outputs/pushbox-small-pred-d8-decoder-*.out` |
| 10 | `pushbox/lewm_small_pred_d10` | `outputs/pushbox-wm-small-pred-d10-*.out` | `outputs/pushbox-small-pred-d10-decoder-*.out` |

The submission entry point is
`scripts/submit_pushbox_predictor_sweep.sh`. The parameterized training scripts
retain depth 6 and `lewm_small` as their defaults, so the original experiment
remains reproducible.

## Primary evaluation

Rank the models using `validate/pred_loss_epoch`, not
`validate/loss_epoch`. The composite validation loss is almost entirely the
weighted SIGReg statistic and can improve even when dynamics prediction gets
worse.

For each depth, record:

1. Best validation prediction loss and its checkpoint.
2. Final validation prediction loss.
3. Mean validation prediction loss over the last 10 epochs.
4. Corresponding training prediction loss and the train/validation gap.
5. Whether the late validation curve is still improving, flat, or degrading.
6. Runtime and model size as capacity costs.

A larger predictor should be considered promising only if it produces a
consistent late-epoch validation improvement. A single best checkpoint is not
sufficient because all variants use one seed and there are no replicate runs to
estimate variance. If depth 8 and depth 10 show the same direction and ordering,
that is stronger evidence than either result alone.

The ultimate criterion is held-out action-conditioned rollout behavior:

- box and gripper position error by rollout horizon
- response to actions rather than static-scene reconstruction
- compounding multi-step drift
- downstream planning success, if available

## Decoder interpretation

Each dependent decoder uses the same frozen projected-latent probe settings:
patch size 14, seed 3072, and 20,000 optimization steps. This provides a
controlled representation-decoding comparison across the jointly trained WMs.

However, the current decoder reconstructs a single observed frame from its
encoder/projector latent. It does **not** decode a latent produced by the
autoregressive predictor. Predictor depth can affect this probe indirectly
because the encoder and predictor are trained jointly, but better decoder MSE
does not by itself demonstrate better prediction.

The automatically submitted probes decode the highest saved checkpoint, which
should be epoch 150 after a successful full run. If validation prediction loss
selects an earlier checkpoint, train an additional decoder for that checkpoint
before making a visual comparison.

For a direct visual predictor test, use a decoder trained in projected-latent
space to decode both:

1. the true next-frame projected latent, and
2. the predictor's next-latent output after `pred_proj`.

Comparing those reconstructions and their rollout degradation would connect
visual quality directly to dynamics prediction.

## Decision guide

| Result | Interpretation | Next action |
| --- | --- | --- |
| Depth 8 improves validation; depth 10 improves further | Strong evidence of predictor under-capacity | Evaluate rollouts and consider depth 10 |
| Depth 8 improves; depth 10 is flat or worse | Capacity helps up to a point | Prefer depth 8 and validate with rollouts |
| Training improves but validation worsens with depth | Overfitting or unpredictable target detail | Add regularization/data; do not increase capacity further |
| Prediction loss is unchanged but decoder improves | Representation is more decodable, not more predictable | Revisit the prediction target/objective |
| Prediction loss improves but decoder is unchanged | Dynamics capacity improved without changing single-frame information | Prioritize rollout evaluation over reconstruction |
| Both training and validation worsen with depth | Not a capacity limit; deeper stack optimizes worse under the fixed recipe | Keep depth 6; retune optimization before any further capacity claim |

## Results (2026-07-22)

All three runs completed 150 epochs. `fit/pred_loss` is the final logged
training-step value; the validation columns come from `validate/pred_loss_epoch`.

| Depth | Predictor params | `fit/pred_loss` | Best val pred (epoch) | Final val pred | Last-20 val mean ± sd | Runtime |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 (baseline) | 24.2M | 0.003255 | **0.004959** (120) | **0.004995** | **0.005031 ± 0.000018** | 3:31:00 |
| 8 | 32.3M | 0.003425 | 0.005592 (123) | 0.005599 | 0.005629 ± 0.000023 | 3:24:08 |
| 10 | 40.4M | 0.003563 | 0.005599 (146) | 0.005614 | 0.005640 ± 0.000023 | 3:55:55 |

Matched projected-latent decoder probes (patch 14, seed 3072, 20k steps, epoch-150
checkpoint):

| Depth | Decoder val MSE | Decoder PSNR |
| ---: | ---: | ---: |
| 6 | **0.02942** | **28.258 dB** |
| 8 | 0.03043 | 28.108 dB |
| 10 | 0.03070 | 28.071 dB |

### Verdict: H1 rejected, H2 rejected, H3 supported

Depth made every measured quantity worse, monotonically:

- Validation prediction loss rose 12.7% at depth 8 and 13.1% at depth 10.
- Training prediction loss also rose, by 5.2% and 9.5%. **This rules out H2.**
  Overfitting requires training loss to improve while validation degrades; here
  the deeper models fit the training data *less* well.
- The decoder probe moved 0.15-0.19 dB the wrong way, so the added depth did not
  buy a more decodable representation either.

The validation gap is far outside within-run noise: late-epoch sd is about
2e-5, while the depth-6 to depth-8 difference is 6e-4, roughly 26 sd. The
train/validation ratio is essentially unchanged across depths (1.53, 1.63,
1.58), which is again inconsistent with a capacity-driven overfitting story.

The remaining explanation is H3 combined with an optimization effect: the
six-layer predictor is not the bottleneck, and at a fixed 150 epochs, lr `5e-5`,
and dropout 0.1 the deeper causal stack simply optimizes worse.

### What this does not establish

- Each depth is a **single seed**. The effect is large relative to epoch-to-epoch
  noise but seed variance was not measured.
- The optimization recipe was deliberately held fixed and therefore was not
  retuned for depth. Deeper transformers typically want warmup or a different
  learning rate, so the honest claim is "depth hurts under this recipe", not
  "predictor depth cannot help".
- No rollout evaluation was run, so nothing here speaks to multi-step drift.

### Follow-up

Keep depth 6. Do not spend further compute widening or deepening the predictor
on the strength of this sweep. The productive next step is the action-conditioned
rollout evaluation already described above, which separates "predicts the next
latent" from "responds to actions" — a distinction none of these one-step numbers
can make.
