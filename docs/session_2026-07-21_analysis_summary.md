# Session summary — 2026-07-21/22: PushBox data coverage, planner diagnosis, real-run forensics

One connected investigation: *why does the real-robot PushBox planner misbehave,
and is it a data problem?* Short answer: the eval harness is clean; the model's
one-step dynamics are fine; failures come from **scene-diverging rollout
hallucination + a flat far-goal latent loss**, both rooted in **data-coverage
gaps** (contact time, push extent) — confirmed end-to-end by the 2026-07-21
live runs.

---

## 1. What was added (all new files; no core training/eval code was modified)

**Analysis scripts** (`scripts/`):

| Script | Purpose |
|---|---|
| `analyze_data_coverage.py` | Dataset coverage audit + pixel-based box tracker (cached `box_track.npz`) |
| `plot_latent_goal_loss.py` | MSE(z_t, z_goal) along every episode video |
| `replay_planner_vs_gt.py` | Offline MPC replay of the deployed planner over validation episodes (sliding/fixed goals) |
| `analyze_planner_vs_gt.py` | Direction confusion matrices + stats from the replay |
| `diag_action_sensitivity.py` | D1: paper's action-shuffle ratio (+negate/zero variants) |
| `diag_token_ranking.py` | D2: 17-token CEM-cost ranking at clear-push states |
| `diag_decode_rollout.py` | D3: decode imagined rollouts at real backward-mistake steps |
| `dryrun_forward_push.py` | Offline planner dry-run on the 28 forward-push start/goal pairs |
| `analyze_real_runs.py` | Forensics on the 2026-07-21 live robot runs |

Plus `README.md`/`INTERPRETATION.md`/`REPORT.md` files in each output folder.

## 2. Analyses, findings, and where they live

### Data coverage — `outputs/coverage/`
Paper-style (hallucination-WM) state/action density audit of
`pushbox_pilot_train.h5` (227 eps / 74.5k steps / 4.1 h). Box pose columns are
**all NaN**, so the box is tracked from pixels (99.9% detection). Findings:
only ~11% of steps move the box; actions are a discrete 8-way keyboard star
(top-15 vectors = 92.5% of steps); yaw never varied; lateral pushes dominate
(strictly-forward is ~7% of pushes; longest forward segment 33 px). Verified
mapping: table +x = down-image.

*Box-tracking QA:* red-patch detector uses a **two-tier threshold** — strict
(R>100, R−G>50, R−B>50, ≥60 px) first, then a looser tier only when strict
finds nothing. The loose tier recovers the washed-out pale-pink patch in the
bright far region of the table (raising detection 98.5% → 99.9%, 21 → 2
episodes below the 95% QA bar) without letting the robot's red parts win;
validated by 0.16 px median centroid agreement where both tiers fire.
Verification montages: `track_check.png` (random / smallest-blob / extreme /
missed frames) and `track_check_recovered.png`.

### Latent goal loss — `outputs/latent_goal_loss/`
The planner's exact goal cost along all 227 episodes: excellent goal/non-goal
contrast (~209×), mostly monotone (median ρ = −0.69), **but nearly flat until
the final ~15% of an episode** — far goals give CEM (horizon 8–10) almost no
signal, and near-goal values are arm-dominated.

### Offline MPC replay vs teleop ground truth — `outputs/planner_vs_gt/`
Full validation set (23 eps × 7,428 replans × 2 goal modes), deployed planner
settings, teacher-forced context. Sliding goal (t+25): 27.2% direction
accuracy (chance 12.5%, κ=0.17), 70.7% within 45°. **Fixed deployment-style
goal: 12.7% vs 11.8% chance (κ=0.01) — indistinguishable from ignoring the
goal.** See `INTERPRETATION.md` there for the panel-by-panel reading guide.

### Backward-action diagnosis — `outputs/diagnostics/`
Motivated by observed backward commands on the robot. A full audit of the
`eval_pushbox_real.sh` chain (sign conventions, feedback loop, planner/training
alignment, CEM sampling) found **no harness bug**. Three-part diagnosis:
- **D1** shuffle ratio 2.12, negate 3.24 → one-step dynamics are
  action-sensitive (not action-marginalized);
- **D2** at clear-push states the cost prefers forward over backward 78%,
  GT token median rank 1/17 → cost ranks well at contact;
- **D3 (smoking gun)** at real 180°-mistakes, imagined horizon-10 futures under
  forward vs backward decode to nearly identical blurry scenes with the box
  vanished, yet costs differ 2–5× → **scene-diverging rollout hallucination**;
  CEM exploits off-manifold latent drift (median chosen plan "predicts"
  closing 61% of goal distance in 2 s).

### Forward-push capability test — `outputs/forward_push/`
28 strictly-forward push segments extracted (all in training episodes — the
validation split contains none), start/goal pairs saved for robot trials.
Offline dry-run with the deployed CEM: **23/28 first actions forward-ish,
25/28 net-forward plans**; misses concentrate at visually distant goals.

### Live-run forensics — `outputs/real_runs_20260721/`
Seven 2026-07-21 robot trials (3 success, 4 fail), identical settings, only the
goal image differed. **Initial latent goal distance separates success from
failure perfectly (≤1.90 vs ≥1.95)**, matching the offline chance-collapse
boundary. Failures are NOT box-position coverage (failures sit in 91–100th
density percentile) but **push-extent coverage**: failed goals demanded
50–55 px displacement vs a 33 px maximum ever demonstrated. One failure
(192115, box 10 px from goal, gd stuck at 2.0) isolates **arm dominance** in
the CLS latent. Stored `z`/`z_hat` latents yield a working contact detector:
one-step surprise 2.2–2.6× baseline in successes vs 0.6–1.2× in failures.

### Pre-existing (2026-07-17), documented this session
`outputs/real_cem_latent_replay_20260717_202906_1203310/` — decoded latent
replay of an earlier live run (`scripts/replay_real_cem_latents.py`).

## 3. Techniques borrowed from `paper/hallucination_wm.pdf`

**Directly applied:**
- *Coverage-as-cause thesis* → framed the whole investigation.
- *State-density coverage maps* (their Fig. 4/6) → `outputs/coverage/`.
- *Action-shuffle ratio* (their eval metric iii, "actions ignored" ≤ 1.1) →
  D1 in `outputs/diagnostics/` (result 2.12 → actions used).
- *Three-mode hallucination taxonomy* (perceptual / action-marginalized /
  scene-diverging) → used to **classify** the backward-action bug as
  scene-diverging (mode iii), the key diagnostic conclusion (D1 rules out mode
  ii, D3 confirms mode iii).
- *Coverage-aware resampling* (their training mitigation) → in recommendations.

**Adapted in spirit (not their exact formulation):**
- One-step surprise ‖ẑ − z_next‖² as a runtime "am I interacting?" signal
  (`outputs/real_runs_20260721/`) — analogous to their runtime predictors.
- D3 decoded-rollout visualization — echoes their qualitative hallucination
  figures, using our trained CLS decoder.

**Their three predictors NOT implemented, and why:** `u_r` (tokenizer
round-trip residual), `u_f` (flow instability), `u_s` (inter-seed variance) are
designed for a Dreamer4 flow-matching + tokenizer model. LeWM is a
deterministic JEPA: no tokenizer round-trip in the loss, no Euler denoising
substeps (`u_f` N/A), no stochastic seeds (`u_s` N/A). Only `u_r` is
adaptable — via encode→decode→re-encode using our separate decoder — and is
left as future work for a runtime hallucination detector.

## 4. Model architecture vs `paper/HierarchicalWM.pdf` (Franka pick-&-place)

Our checkpoint `weights_epoch_146.pt` totals **18.04 M params**: predictor
10.79 M, encoder 5.50 M, projector 0.80 M, pred_proj 0.80 M, action_encoder
0.16 M.

| | Yours (LeWM PushBox) | HWM Franka (low-level WM = VJEPA2-AC) |
|---|---|---|
| Total WM params | **18.0 M** | **~1.3 B** (frozen ViT-g/16 ~1 B + ~300 M WM) |
| Encoder | ViT-tiny (192-d, patch 14), **from scratch** on 4 h | Pretrained **frozen** ViT-g/16 |
| Latent per frame | **single 192-D CLS token** | **full spatial patch-feature map** |
| Predictor | causal AdaLN transformer, context 3 | ~300 M ViT, context 16, T=15 windows |
| Proprioception | **not in the model** (pixels + action only) | EE pose ∈ ℝ⁷ as an input token |
| Action space | 2-D XY delta, ≤10 mm | 7-D EE delta pose (pos+Euler+gripper) |
| Training loss | one-step TF latent **L2** + SIGReg; **no rollout loss** | TF **L1** + **multi-step rollout loss** (γ_tf=γ_roll=1) |
| Data | 4.1 h, one task/scene/camera | ~130 h (96 h DROID + 30 h RoboSet), many scenes |
| Planner (flat CEM) | 1024 samp / 10 iter / 32 elite / horizon 10, terminal **MSE** | 2400 samp / 15 iter / 20 elite, **Var EMA 0.75**, horizon **6**, **L1** cost |

Four gaps map onto our diagnosis: (1) **no rollout loss** — theirs exists
specifically to fight the compounding drift our D3 shows; (2) **from-scratch
tiny encoder** co-adapting vs a frozen billion-param anchor; (3) **192-D CLS
bottleneck** must cram arm+box+background into one vector (→ arm dominance,
finding in `outputs/latent_goal_loss/` and live 192115); (4) **no proprio**
though our dataset already logs it. Scale caveat: 18 M/4 h vs 1.3 B/130 h.
Encouraging: their *flat* planner also scores **0%** on non-greedy tasks — flat
latent MPC is fragile at any scale — but our forward-push test is **greedy**,
the regime where flat planners *do* work (they solved greedy drawer variants).

## 5. Discussions / decisions

- **HWM code feasibility** (github.com/kevinghst/HWM_PLDM): public repo is the
  PLDM/maze instantiation only. Recommended path: (1) add rollout loss (+
  optional proprio head) to our `train.py` — cheapest, targets the diagnosis;
  (2) adopt DINO-WM-style frozen-DINOv2 spatial latents (public repo, fits
  16 GB GPU); hierarchy only later — our failures are low-level fidelity, not
  long-horizon decomposition.
- **Cheap no-retraining CEM/cost tweaks** (from their Tables 8/11–12, testable
  immediately): **L1** goal cost instead of MSE (less dominated by a few large
  off-manifold latent coords — directly targets the D3 exploit); **Var-EMA
  ≈0.75** smoothing on the CEM std to avoid premature collapse; shorter
  **horizon ~6** (D3 says horizon-10 rollouts are where plans diverge).
- **Goal-selection protocol for future trials:** goal gd ≲ 1.9 and box
  displacement ≲ 30 px (both checkable before running); chain subgoals for
  longer pushes; take goal photos with the arm in a canonical retracted pose.
- **Next data collection:** long continuous pushes, more contact time, box
  rotations, coverage-aware resampling of contact-rich windows.

## 6. Open follow-ups (suggested, not yet run)

- **Loss vs box-pixel-distance re-plot:** re-plot the latent goal loss against
  box→goal *pixel* distance (from `box_track.npz`) instead of episode time, to
  cleanly separate "loss tracks the box" from "loss tracks the arm."
- **Arm-in vs arm-out goal probe:** compare goal images with the arm in vs out
  of frame to quantify how much the CLS latent's goal distance is driven by arm
  pose rather than box state (the suspected 192115 failure mechanism).
- **`u_r` runtime detector:** adapt the paper's tokenizer round-trip residual
  via encode→decode→re-encode using our CLS decoder, as a live hallucination
  signal on the robot (see §3).
