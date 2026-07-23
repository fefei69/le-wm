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
