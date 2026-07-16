# Real-World PushBox Experiment Recommendations

This note records the recommended world-model checkpoints and decoder experiments
for the real-world PushBox evaluation. The recommendations are based on the
150-epoch PushBox LeWM run and the final CLS-decoder reconstruction grid.

## World-model checkpoint selection

Use validation prediction loss (`validate/pred_loss_epoch`) as the primary
checkpoint-selection signal. Do not rank checkpoints by
`validate/loss_epoch`: that value is almost entirely the stochastic SIGReg
term, and its minimum occurs at a checkpoint with substantially worse dynamics
prediction.

The saved checkpoint number is one greater than the zero-based W&B epoch. The
best candidates are:

| Checkpoint | Validation prediction loss | Difference from best |
| --- | ---: | ---: |
| `weights_epoch_146.pt` | **0.00458957** | best |
| `weights_epoch_140.pt` | 0.00459291 | +0.073% |
| `weights_epoch_148.pt` | 0.00459639 | +0.149% |
| `weights_epoch_150.pt` | 0.00460009 | +0.229% |
| `weights_epoch_129.pt` | 0.00460107 | +0.251% |

### Recommended test set

1. Use `weights_epoch_146.pt` as the primary model.
2. If three checkpoints can be evaluated, test epochs **140, 146, and 150**.
   This tests the late-training plateau without spending evaluation time on
   nearly identical adjacent epochs.
3. Add epoch **129** only if an earlier, cheaper checkpoint is useful. Its
   validation prediction loss is already within 0.251% of the best value.

There is no clear validation overfitting in this run. Prediction loss continues
to improve slowly through the late epochs, but differences within the final
plateau are too small to establish the best real-world rollout model from loss
alone. Select the final model using held-out episode rollouts, with emphasis on
box and pusher position, action response, and multi-step drift.

For a fair comparison, train or fine-tune a separate decoder for each tested
world-model checkpoint. Do not use the epoch-150 decoder to rank epoch 140 or
146: latent coordinates can change during training even when the loss is nearly
unchanged. At minimum, the final projected-latent decoder must be trained using
the same world-model checkpoint that will be deployed.

## Decoder diagnosis

The current reconstruction has the correct scene layout but is blurry and shows
a regular patch grid. This is expected from the current probe for three reasons:

- It reconstructs the entire 224x224 image from one 192-dimensional CLS token,
  which may not retain texture and exact spatial detail.
- The decoder predicts independent 16x16 RGB patches with a linear head and
  concatenates them without convolutional refinement across patch boundaries.
- Pixel MSE favors the conditional mean when details are uncertain, which
  produces blur around the robot and box.

The encoder patch size is 14 while the decoder output patch size defaults to 16.
This mismatch is not inherently invalid because the decoder receives only a
global token, but smaller output patches are a useful diagnostic for the visible
grid.

## Recommended decoder experiments

### 1. Isolate the patch artifact

Keep the current decoder and loss fixed, then compare output patch sizes **16,
14, and 8**. Patch size 14 gives a 16x16 output grid matching the encoder's
spatial grid; patch size 8 gives a finer 28x28 grid. Use the same data split,
training steps, seed, and selected world-model checkpoint for all runs.

Expected result: smaller patches should reduce the size and visibility of the
grid cells, but will not remove blur caused by the global latent bottleneck.

### 2. Replace independent RGB patches with spatial refinement

For the deployable decoder, project the latent into a learned low-resolution
feature grid and allow neighboring locations to communicate. Add either query
self-attention or spatial residual blocks, followed by overlapping 3x3
convolutions and upsampling to 224x224. Avoid making a single independent linear
RGB prediction for every output patch.

This is the highest-priority architectural change because it directly removes
uncoordinated patch boundaries while keeping the decoder compatible with a
single predicted planning latent.

### 3. Train in the deployed latent space

The autoregressive world model predicts the projected planning latent, not the
pre-projector CLS token. Train the simulator decoder with `--latent proj` using
the selected world-model checkpoint. Keep the CLS decoder only as a
representation/reconstruction baseline.

The first recommended run is therefore equivalent to:

```bash
python train_decoder.py \
  --checkpoint pushbox/lewm/weights_epoch_146.pt \
  --dataset pushbox_pilot_train \
  --latent proj \
  --patch-size 14 \
  --steps 20000 \
  --out "$STABLEWM_HOME/checkpoints/pushbox/lewm/decoder_proj_ep146_p14"
```

Then compare it with an otherwise identical `--patch-size 8` run and with the
spatially refined decoder.

### 4. Treat full spatial-token decoding as an upper bound

As a diagnostic, decode all encoder patch tokens rather than only CLS. This
should show how much sharp spatial information exists in the encoder but is
discarded by the global bottleneck. It is not directly usable for autoregressive
rollouts because the current predictor produces only the global projected
latent. Report it separately rather than comparing it as a deployable decoder.

### 5. Adjust the image loss after fixing the architecture

Retain MSE or L1 as a stable reconstruction term. If edges remain overly smooth,
add a modest perceptual or edge loss and evaluate both pixel fidelity and
geometry. A perceptual loss may improve apparent sharpness, but it cannot recover
information absent from the latent and must not be allowed to hallucinate box or
pusher positions.

## Evaluation protocol

Use an episode-atomic held-out split so nearly identical adjacent frames do not
appear in both decoder training and validation. For every candidate, record:

- validation MSE and PSNR;
- a patch-boundary or edge-continuity metric;
- box-center and pusher-center pixel error, if annotations or a detector are
  available;
- qualitative grids with identical held-out frames;
- one-step and multi-step action-conditioned rollout error;
- temporal consistency and drift over the rollout horizon.

The real-world choice should prioritize correct object geometry and response to
actions over photographic sharpness. A visually sharper decoder that invents
edges or shifts the box is worse for world-model evaluation than a slightly
blurry but geometrically faithful decoder.
