# PushBox DINO-WM learning curves

## Objective

Train the highest-probability real-world baseline before investing further in
the scratch LeWM architecture:

1. Freeze pretrained DINOv2-small and retain its 16 x 16 grid of patch tokens.
2. Condition dynamics on the recorded 2-D command and 4-D proprioception.
3. Use the published small DINO-WM predictor shape (depth 6, 16 heads,
   2048-wide MLP; roughly the 25M scale).
4. Measure nested 1 h, 2 h, and full-data learning curves.

The experiment asks whether pretrained spatial features plus robot state can
learn useful xAI-specific local dynamics from the existing recording.

## Controlled setup

All curves use:

- the same deterministic episode-level validation split;
- the same shuffled episode order, making the 1 h set a subset of 2 h and the
  2 h set a subset of the full-data run;
- train-subset-only action and proprio normalization;
- 224 x 224 images, 14-pixel patches, history 3, horizon 1, frame skip 1;
- frozen `facebook/dinov2-small` features;
- 10 epochs, batch size 32, AdamW at 5e-4.

Because validation remains disjoint and fixed, the nominal 4 h curve uses all
episodes left in the training split, about 3.7 actual hours. Each checkpoint
directory records exact episodes, clips, duration, normalization statistics,
and parameter counts in `split_manifest.json`. Metrics are logged locally as
CSV by default; W&B remains disabled unless the user explicitly enables it.

## Run

```bash
bash scripts/submit_pushbox_dinowm_curves.sh
```

Individual curves can be submitted with, for example:

```bash
bash scripts/submit_pushbox_dinowm_curves.sh 1
```

Outputs are isolated under:

```text
$STABLEWM_HOME/checkpoints/pushbox/dinowm_dinov2s_prop_1h/
$STABLEWM_HOME/checkpoints/pushbox/dinowm_dinov2s_prop_2h/
$STABLEWM_HOME/checkpoints/pushbox/dinowm_dinov2s_prop_4h/
```

## Primary readout

Compare `validate/loss_epoch`, `validate/pixels_loss_epoch`, and
`validate/proprio_loss_epoch` at matched epochs. A consistently improving
1 h -> 2 h -> 4 h curve supports collecting more targeted xAI data. A plateau
by 2 h shifts attention to rollout evaluation, action sensitivity, predictor
capacity, or missing state rather than raw duration.

The corresponding `metrics.csv` files live below each run's `logs/csv/`
directory under `$STABLEWM_HOME/checkpoints/`.

One-step feature loss is necessary but not sufficient. Before using the model
for control, follow with correct-action versus shuffled-action evaluation and
recorded-trajectory rollouts at 1, 5, 10, and 25 steps.

## Post-hoc pixel decoders

Train the non-quantized, VQ-VAE-style pixel decoder for every completed curve:

```bash
bash scripts/submit_pushbox_dinowm_decoders.sh
```

The DINO backbone and world-model predictor remain frozen. Each decoder uses
the exact train episodes in its curve's `split_manifest.json` and the common
held-out validation episodes. Decoder MSE, PSNR, and reconstruction grids are
logged to the `pushbox-dinowm-decoders` W&B group and written below:

```text
$STABLEWM_HOME/checkpoints/pushbox/dinowm_dinov2s_prop_1h/decoder_vqvae/
$STABLEWM_HOME/checkpoints/pushbox/dinowm_dinov2s_prop_2h/decoder_vqvae/
$STABLEWM_HOME/checkpoints/pushbox/dinowm_dinov2s_prop_4h/decoder_vqvae/
```

The original DINO-WM decoder maps a 14 x 14 token grid to 224 x 224 pixels.
These checkpoints contain a 16 x 16 grid, so `DinoPatchDecoder` retains all
256 tokens, produces its native 256 x 256 output, and performs one final
bilinear resize to 224 x 224.

## Results (2026-07-22)

All three curves completed 10 epochs against the same held-out 23 episodes
(0.414 h, 7,382 clips). Every run has 42.3M total parameters, of which 22.1M are
the frozen DINOv2-small backbone and 20.2M are trainable.

| Curve | Actual train | Episodes | Clips | `validate/pixels_loss_epoch` | Change | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1h | 1.002 h | 54 | 17,883 | 0.25642 | — | 0:48:52 |
| 2h | 2.000 h | 108 | 35,678 | 0.22806 | -11.1% | 0:58:42 |
| 4h | 3.727 h | 204 | 66,468 | **0.21748** | -4.6% | 1:07:24 |

Normalized per doubling of data, the return roughly halved: -11.1% per doubling
from 1 h to 2 h, then -5.2% per doubling from 2 h to 3.727 h. Total improvement
across the full 3.7x data range is -15.2%.

### Read the pixel loss, not the composite or proprio loss

Only `validate/pixels_loss_epoch` is comparable across these three runs.

- **`pixels_loss` is valid.** Its target is the frozen DINOv2-small patch
  embedding, which is byte-identical in all three runs, so the metric measures
  the same quantity on the same validation episodes.
- **`proprio_loss` is not comparable and should be ignored across curves.** The
  target is `proprio_emb`, the output of a jointly trained 10-dimensional
  `Embedder`, so each run measures error in its own learned space at its own
  arbitrary scale. It sits on top of a second confound: proprio is z-scored with
  train-subset-only statistics, and those differ per curve (per-dimension std
  spreads about 3.5% across subsets). The apparent monotone rise of
  `proprio_loss` (0.01346 -> 0.01473 -> 0.01620) is therefore not evidence of
  anything about data scaling.
- **`loss` is dominated by pixels** — it is a plain MSE over the concatenated
  actionless embedding, where 384 pixel dimensions per token swamp the 10
  proprio dimensions — so it tracks `pixels_loss` and adds no information.

### Decoder results measure the feature ceiling, not the curves

| Curve | Decoder val MSE | Decoder PSNR |
| --- | ---: | ---: |
| 1h | 0.00495 | 35.98 dB |
| 2h | 0.00455 | 36.35 dB |
| 4h | **0.00396** | **36.94 dB** |

These numbers do **not** rank the world models. `train_dino_decoder.py` loads
only the checkpoint's DINO backbone and decodes ground-truth frozen patch
tokens, and that backbone was never trained, so all three decoders reconstruct
from *identical* features. The 0.96 dB spread reflects how much data each
decoder itself was trained on, nothing about the predictors.

What the table does establish is a useful reference ceiling: frozen
DINOv2-small patch tokens support roughly 36-37 dB reconstruction on PushBox,
against 28.3 dB for the best scratch-LeWM projected-latent probe. That gap is
mostly an information-budget difference — 256 x 384 tokens versus a single
192-dimensional vector — so it bounds what the LeWM probe could reach rather
than proving DINO features are better for dynamics.

### Interpretation

The curve is monotonically improving but not scaling strongly: the marginal
return per doubling halved between the two steps, projecting to roughly 3-4% for
a further 4 h -> 8 h doubling. This is neither the clean plateau nor the strong
scaling signal the primary readout was written to distinguish.

More data helps, but weakly enough that another doubling is a poor use of
collection effort compared with the evaluation gap. One-step feature loss cannot
tell whether the model responds to actions at all. Run the correct-action versus
shuffled-action comparison and the 1/5/10/25-step recorded-trajectory rollouts
before deciding anything about additional xAI data collection.
