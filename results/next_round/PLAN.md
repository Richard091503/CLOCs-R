# Next round: pre-registered rules

Each phase's decision rule is written here before that phase is run, with its date, and is not edited afterwards.
Changes are added as dated addenda at the end of the phase's section.

## Phase 0: environment, data and reproduction (written 2026-11-17, before any Phase 0 fusion or detector result)

**Setting.** Jetstream2 instance, GRID A100X-10C (10 GB slice of an A100, sm_80), 8 CPUs, 29 GB RAM, 58 GB disk.

**Deviations from the RTX 4090 run of `results/full_dataset/`, decided before any result:**

- **Data route.** Kaggle returns HTTP 404 to every request from this host, so the dataset was converted with
  `fetch_aimotive.py --src ... --train_stride 1` on another machine and copied here (9.4 GB). Frame counts match
  `results/full_dataset/METHODS.md` exactly in every split and condition.
- **Detectors are retrained here**, not copied: SECOND and Cascade R-CNN with their committed configurations. This was
  decided before any result. So detector variance, which the reported seed ranges do not include, enters this comparison.
- **Cascade R-CNN** is trained with batch 2 and 2-step gradient accumulation (`cascade_2d.py --accum 2`), because batch 4
  at 1.5x does not fit in 10 GB. Each optimizer step still sees 4 images, warmup is kept at 300 optimizer steps, and the
  backbone's BatchNorm statistics are frozen (`norm_eval`), so this is equivalent up to sample order.
- **SECOND** uses 6 data-loader workers instead of 12 (8 CPUs), which changes the augmentation random streams.
- **Detectors are trained one after the other** instead of at the same time.

**What is run.** With the retrained detectors, exactly the steps of `clocs_r/run_full.sh` that produce the inputs
(candidates, detectors alone, fusion cells and radar features with gate δ0 = 0.5 m, κ = 0.02, 3 sweeps), then the
final pair, R2+T2 and R0+T2, trained with `tune_fusion` (far weight 2, 3 epochs) for seeds 1–6 and scored on holdout and
val. Selection is not rerun: the configurations are fixed by `results/full_dataset/ablation/decisions.json`.
As a side check, the committed 4090 fusion networks (`results/full_dataset/weights/`) are also scored on the new
candidates; they were trained on the old detectors' outputs, so they are not expected to reproduce.

**Reference values** (val, center-distance AP beyond 50 m, 6 seeds, from `results/full_dataset/ablation/val.json` and
`extra_seeds.json`):

| Model | Mean | SD (ddof 1) |
|---|---|---|
| SECOND alone | 0.453 | single run |
| R0+T2 | 0.4145 | 0.0080 |
| R2+T2 | 0.4224 | 0.0081 |
| Radar gain (R2+T2 − R0+T2, seed against seed) | +0.0079 (−0.0030 to +0.0245), 5 of 6 positive | |

**Rule.** For each of R0+T2 and R2+T2, the run **reproduces** if the new 6-seed mean on val beyond 50 m differs from the
reference mean by at most 2 standard errors of the difference, `2 · sqrt(sd_ref² / 6 + sd_new² / 6)`. Both must
reproduce for Phase 0 to pass. SECOND alone and the radar gain are reported against their reference values with no
pass/fail threshold, because SECOND was trained once in each run, and the gain's sign was already mixed across seeds.
If the rule fails, nothing is retuned. The difference is reported with SECOND's own change, since a different SECOND
shifts every fused model.

## Phase 1: a selection split (written 2026-11-18, before the split was chosen; only training labels were counted)

**What was known when this was written.** Scored-car counts (car, van and pickup in the region and in camera view) per
training scene and per recording, from the labels alone (`runs/next_round/phase1/scene_stats.json`). No detector or
fusion output was looked at on any candidate split.

**Why a new split.** The current holdout has no rain cars beyond 50 m and 13 night cars at 100–200 m. It also takes
every 10th scene, so most holdout scenes sit next to training scenes from the same drive (15 s apart).

**Rule for choosing it.**

- The unit is a whole recording (date-time and vehicle, e.g. `20210723-222803@Yoda`), so no drive is in both parts.
- Per condition, the chosen recordings are the subset of 2 to 4 recordings whose share of that condition's training
  frames is closest to 15% and whose share of its cars at 100–200 m is closest to 20%. The score is
  `|frames share − 0.15| + |far share − 0.20|`, and ties go to the subset with fewer recordings, then the alphabetically
  first.
- Rain has only 2 training recordings, so it gets the one with cars beyond 100 m, by the same score.
- The training recording that also appears in val (`20210728-063719@Yoda`, urban) is never chosen.
- The split is named `select`. The rest of aiMotive's training scenes, including the old holdout scenes outside the
  chosen recordings, are named `fit`. The existing `train`, `holdout` and `val` are unchanged.

**Not decided here.** Whether the detectors are retrained on `fit` is not decided here, since that is the user's call.
The current detectors were trained on every training scene except the old holdout, so they have seen most of `select`.
Nothing is trained or selected on `select` in this phase.

### Phase 1 addendum (2026-11-18, before any detector was trained on `fit` and before any fusion run on `select`)

**What was found.** Scored on `select` and `train`, where they were trained, the Phase 0 detectors are far better than
on unseen scenes:

- SECOND, APCD beyond 50 m: 0.728 on `train` and 0.804 on `select`, against 0.434 on holdout and 0.457 on val.
- Cascade R-CNN, candidate recall@0.5 at 100–200 m: 0.918 on `train`, against 0.226 on val.

So `select` needs detectors trained without it. Fusion trained on `train` also learns from in-sample detector outputs,
which are much better than what it meets on val. The user approved doing what is best for the project.

**Detectors.** SECOND and Cascade R-CNN are retrained on `fit`, with Phase 0's settings (`--workers 6`, `--accum 2`).
These are the detectors that score `select` and val from now on.

**Cross-fitted training inputs.** `fit` is split into 3 folds of whole recordings (`splits.FOLDS`, `fold0`–`fold2`).
For each fold K, both detectors are trained on `notfoldK` with the same settings and export candidates only for `foldK`.
Fusion trained on these cross-fitted candidates sees detector outputs on scenes the detectors never saw. Fusion trained
on the `fit` detectors' own outputs on `fit` is the in-sample arm, which is the current pipeline. `select` and val
always use the `fit` detectors.

**What is run on `select`.**

- R0+T2 and R2+T2, far weight 2, gate 0.5 m and 0.02, 3 epochs, seeds 1–3, trained on `fit`, once per arm (in-sample
  and cross-fitted).
- SECOND and Cascade R-CNN alone, from the `fit` detectors.

**Rule.**

- The cross-fitted arm is adopted if R2+T2's 3-seed mean APCD beyond 50 m on `select` is higher than the in-sample
  arm's by more than the standard deviation (ddof 1) of the cross-fitted arm's 3 seeds. Otherwise the in-sample arm is
  kept.
- R0+T2 follows the same choice, so the radar comparison stays matched.
- Its results under both arms are reported.

**Val, once.** For the chosen arm, seeds 4–6 of R0+T2 and R2+T2 are trained, and all 6 seeds are scored on `select`
and val. SECOND alone is scored with the `fit` detectors. These become the reference for later phases. The Phase 0
numbers are reported alongside them, but are not comparable as a test, since the detectors differ.

## Phase 2: the range-switch baseline (written 2026-11-19, before it was computed)

**What is computed.** Nothing is trained or chosen.

- For each candidate, the score is R2+T2's fused score if the candidate's BEV range is under 50 m, and SECOND's own
  score otherwise. Then the same BEV NMS (0.1) and evaluation as every fused model follow.
- R2+T2 is the 6 networks scored on val in Phase 1 (`runs/next_round/phase1/fusion_insample`, the `fit` detectors).
- The same switch is also computed with R0+T2. The switch is scored on `select` and val.
- Beyond 50 m it equals SECOND by construction. The only differences come from NMS between boxes on either side of
  50 m, and are reported.

**Standing rule for later phases.** Phase 1 showed that `select`'s pooled 50–200 m bin is 71% cars at 50–100 m, against
50% on val.

- From Phase 3 on, every rule that compares models beyond 50 m is applied both to the pooled 50–200 m bin and to the
  100–200 m bin on its own.
- A model "beats the range switch" only if its mean on `select` is higher in both bins.
- The margin in each bin must be more than the standard deviation of the model's seeds.

**Phase 2 addendum (2026-11-19, after it was computed).**

- Beyond 50 m the switch differs from SECOND for a second reason besides boundary overlap: the NMS caps in
  `metrics.nms_bev`, which keep the top 1,000 boxes before NMS and 100 after. Changing the scores inside 50 m changes
  which far boxes make those cuts.
- The bar for Phase 3 on `select` is switch(R2+T2): 0.562 at 50–200 m and 0.405 at 100–200 m. The standing rule is
  applied against these numbers.
- The NMS settings stay as they are for every model, so all comparisons use the same caps.


## Phase 3: tell the network when the camera can see (written 2026-11-19, before anything was run)

**Base.** R2+T2 as in Phase 1: `fit` detectors, in-sample cells (`runs/next_round/phase1/cells_in`), far weight 2, gate
0.5 m / 0.02, 3 sweeps, 3 epochs. With no new flag it must reproduce Phase 1's `select` numbers. It is retrained
here, seeds 1–3, as a check, and that retrained run is the reference.

**Changes**, each a token added to the configuration name, each behind a flag (`clocs_r/visibility.py`,
`train_fusion.fit`):

| Token | Change |
|---|---|
| H | Projected 2D height of the 3D candidate: log(1 + pixels) / log(1 + image height); −1 if not projectable. One more channel in every cell. |
| P | LiDAR points inside the 3D box: log(1 + n) / 8. One more channel. |
| I1, I2 | A 2D-3D pair is a cell only if IoU > 0.1 (I1) or > 0.25 (I2), instead of > 0. Candidates left without a pair use the no-camera row. |
| K40 | Only each frame's 40 highest-scoring 2D candidates form pairs. |
| D2, D3 | Camera dropout in training: all 2D candidates removed for a random 20% (D2) or 30% (D3) of training steps. |
| S50, S100 | Frames with a car beyond 50 m (S50) or 100 m (S100) are drawn twice as often in training. The number of steps is unchanged. |

On `fit`, 86% of frames have a car beyond 50 m and 34% have one beyond 100 m, so S50 changes little. S100 is the variant
aimed at 100–200 m, where fusion loses to SECOND.

**Rule.** "Beats X" means the mean over seeds 1–3 on `select` is higher than X's mean by more than the standard
deviation (ddof 1) of those seeds. This must hold both at 50–200 m and at 100–200 m, with APCD (the standing rule).

1. **Stage A.** Each token alone is added to R2+T2. A token passes if R2+T2+token beats the reference. Within each
   group of alternatives ({I1, I2, K40}, {D2, D3}, {S50, S100}), only the passing one with the higher 50–200 m mean is
   kept.
2. **Stage B.** The kept tokens are ordered by their 50–200 m gain in Stage A, largest first. Start from R2+T2 and add
   them one at a time. A token is adopted if the configuration with it beats the configuration without it. That needs
   one run of 3 seeds per token after the first.
3. **Final.** The configuration after Stage B, plus its counterpart without radar (R0 in place of R2, same tokens), each
   with seeds 1–6. Both are scored once on `select` and once on val, with SECOND, the switch, R0+T2 and R2+T2 beside
   them. If nothing is adopted, the final is R2+T2, already scored on val in Phase 1, and val is not scored again.
4. **The switch.** Every Stage A, Stage B and final configuration is also checked against switch(R2+T2) on `select`
   (0.562 at 50–200 m, 0.405 at 100–200 m) by the same rule. The report says whether any of them beats it.

Nothing in this rule depends on val. The parameter count stays in the low thousands: each new channel adds 18
parameters.

## Phase 4: gated fusion (written 2026-11-20, before anything was run)

**Model** (`fusion.GatedFusion`, token G): `s_hat = logit(s3D) + g_cam * cam_term + g_radar * radar_term`.

- `cam_term` is the max over a candidate's cells of `f_cam(IoU, s2D, s3D, distance)`. This is the CLOCs layer stack,
  4 → 18 → 36 → 36 → 1. The no-camera row is one of the cells.
- `radar_term` is `f_radar(radar features of the set, distance, s3D)`, once per candidate: 3 → 16 → 16 → 1 for R2.
- `(g_cam, g_radar) = sigmoid(f_gate(context))`: c → 16 → 2. The context is:
  - projected 2D height and LiDAR point count (Phase 3 H and P);
  - log(1 + radar returns in the gate) and log(1 + moving returns), where moving means |ego-compensated radial
    speed| ≥ 1 m/s;
  - distance.
- **Token C** adds a one-hot driving condition (highway, urban, night, rain) to the context. The plan asks for both
  with and without it, so both are trained.
- The last layers of `f_cam` and `f_radar` start at zero, so training starts from SECOND's ranking. The gates start
  at 0.5.
- About 2.6k parameters.
- Training changes: K40, the only one Phase 3 adopted. Far weight 2, 3 epochs, the same seeds.
- **Without radar** (R0 in place of R2) there is no `radar_term`, and the context drops the two radar counts. This
  is the counterpart for the radar gain.

**Reference.** R2+T2+K40 seeds 1–3 from Phase 3 (`runs/next_round/phase3/fusion`).

**Rule.** "Beats" is as in Phase 3: the mean on `select`, seeds 1–3, is ahead by more than the candidate's seed SD,
both at 50–200 m and at 100–200 m (APCD).

1. A candidate is R2+T2+K40+G or R2+T2+K40+G+C. It passes if it beats the reference.
2. **Camera-removal veto** (new; motivated by Phase 3, where the camera detector is much better on `select` than on
   val beyond 100 m).
   - The candidate and the reference are also scored on `select` with every 2D candidate removed (`--nocam`).
   - A candidate that passed step 1 is vetoed if, under camera removal, its mean is below the reference's by more than
     its own seed SD in either bin.
   - The veto can only remove a candidate, never add one.
3. If both pass, the one with the higher 50–200 m mean on `select` is the final.
4. **Final:** it and its R0 counterpart, seeds 1–6, scored once on `select` and once on val.
   - If neither passes, the final remains R2+T2+K40 and val is not scored again.
   - In that case the gate plot is made on `select`.
5. Every candidate is also checked against switch(R2+T2) on `select` (0.562, 0.405), and the report says whether it
   beats it.

**Gate plot** (`results/next_round/figures/gates.png`): mean `g_cam` and `g_radar` by distance bin (0–30, 30–50,
50–100, 100–200 m) and condition.

- Candidates are SECOND score ≥ 0.05, in the scored region.
- Shown separately for true cars (center within 2 m of a scored car) and the rest.
- Made on `select` for both G variants, and on val for the final if it is gated.

## Phase 5: the ceiling of a radar detector (written 2026-11-20, before any Phase 5 run)

A measurement, not a selection: no network is trained, and nothing is tuned. Code: `clocs_r/oracle_inject.py`.

**Target cars.** Scored cars on `select` and val that meet all of the following:
- They are in the scored region and the camera view, as in `cache_cells.eval_labels`.
- They are at least 50 m away (BEV range).
- They have **no LiDAR point** inside the box. The points are the single sweep SECOND sees (`seq.lidar`), counted with
  `visibility.points_in_boxes`.
- They have **at least one radar return in the gate**: 3 sweeps, δ0 = 0.5 m, κ = 0.02, the same gate as the radar
  features.

**Counts.** Reported by split, condition (highway, urban, night, rain, plus overall) and bin (50–100, 100–200 m):
- every scored car beyond 50 m;
- those with no LiDAR point;
- those with no LiDAR point and a radar return (the targets);
- the targets SECOND already finds: a SECOND candidate with score ≥ 0.05 within 2 m of the center.

**Injection.**
- Each target's ground-truth box is added to its frame as one more 3D candidate, with s3D = **0.1**.
- All the inputs a SECOND candidate has are built the same way: 2D pairs, radar features, projected height,
  LiDAR points (0) and moving returns. The K40 pairing applies to the injected candidates too.
- Every model then scores the frame as usual, followed by the usual NMS.
- **Low** (the headline, as the plan asks): the injected candidates keep whatever score each model gives them.
- **Perfect**: the injected candidates' final score is set to 1 before NMS. This is the absolute ceiling.
- Each model is also scored without injection, from the same forward pass. Scoring is per candidate, so this must
  reproduce the earlier numbers exactly; the report checks it.

**Models.** All from saved weights, seeds 1–6:
- SECOND alone;
- switch(R2+T2), at 50 m;
- R0+T2 and R2+T2 (`runs/next_round/phase1/fusion_insample`);
- R2+T2+K40 (`runs/next_round/phase3/fusion`);
- R0+T2+K40+G and R2+T2+K40+G (`runs/next_round/phase4/fusion`). R2+T2+K40+G is the Phase 4 final.

**Ceiling.** The injected APCD minus the APCD without injection, seed against seed, in every bin and condition.
APCD beyond 50 m is the headline. The report also gives how many injected candidates survive NMS.

**Recommendation rule for Phase 6.**
- A radar detector is worth building if the **low** ceiling of R2+T2+K40+G at 50–200 m on `select` is at least
  **+0.010 APCD**, as the mean of 6 seeds.
- 0.010 is above the largest gain any change in Phases 2–4 made on `select` at 50–200 m (+0.0074, token G). A detector
  that is perfect on these cars should at least clear that.
- val is scored once, in the same run, and reported beside it.
- The decision to start Phase 6 stays with the user.

### Addendum to Phase 5 (2026-11-20): the recommendation was overridden

- By the Phase 5 rule, a radar detector was not worth building. The low ceiling of R2+T2+K40+G at 50–200 m on
  `select` was +0.0070, below +0.010.
- The user decided to run Phase 6 anyway, after seeing that val's ceiling is about 2.7× larger (+0.019 low, +0.058
  perfect).
- `select` has no target cars in rain and 23 at night. val has 131 and 99.
- The Phase 5 rule and its outcome stand as written. Phase 6 runs by the user's decision, not by that rule.

## Phase 6: a radar detector as a second candidate source (written 2026-11-20, before any Phase 6 run)

**1. The radar detector.**
- **Model and code:** OpenPCDet PointPillars on radar points only. Config `clocs_r/configs/pointpillar_radar_aimotive.yaml`;
  dataset `clocs_r/radar_dataset.py`; trained with `train_second.py --config`.
- **Points:** 5 sweeps accumulated into the current frame's body coordinates, with no Doppler motion correction.
  Per-point features:
  - x, y, z;
  - Doppler speed with ego motion removed (`Sequence.radar_sweep`), divided by 40 m/s;
  - log RCS, scaled as `radar_features.RCS_LOG_RANGE`;
  - sweep age in seconds.
- **Grid:** the same x and y range as SECOND (0–204.8 m, ±25.6 m). z is −5 to 5 m. Pillars are 0.32 × 0.32 m (radar
  positions are much coarser than LiDAR's), with up to 10 points per pillar.
- **Labels:** cars in the camera view, as for SECOND, kept only if they have at least one radar return in the gate
  (δ0 0.5 m, κ 0.02, 5 sweeps). For SECOND the equivalent was at least one LiDAR point.
- **Anchors:** the same Car anchors and range-weighted head as SECOND.
- **Training:** trained on `fit`, as SECOND is, for 40 epochs. Batch size, learning rate and augmentation are as for
  SECOND. Only the last epoch is used.
- **Export:** the boxes after BEV NMS (0.1, at most 100 per frame) with score ≥ 0.01, for `fit`, `select` and val.
- **Reported on its own:** APCD and the other `metrics.py` metrics by distance bin and condition, on `select` and val.

**2. Integration** (`clocs_r/radar_candidates.py`, token **RD** in `tune_fusion`).
- Radar detections are the exported boxes with score ≥ 0.05.
- **Matched:** each SECOND candidate gets a channel `rd`, the highest score of any radar detection whose BEV center is
  within 2 m of it, or 0 if there is none.
- **Unmatched:** a radar detection with no SECOND candidate of score ≥ 0.05 within 2 m becomes a new candidate. Its
  inputs:
  - s3D = its radar score;
  - `rd` = the same score;
  - source flag `src` = 1 (0 for SECOND candidates);
  - its own radar features, projected height, LiDAR points and moving returns;
  - its 2D pairs, built as `cache_cells.base_item` does.
- RD appends `rd` and `src` to the radar channels, and `src` to the gate context of the gated model.
- **Training targets for radar-sourced candidates:** 1 if the BEV center is within 2 m of a scored car; weight 0
  between 2 and 4 m and on trucks and buses; 0 otherwise. SECOND candidates keep their targets.
- **Deduplication:** across sources by the usual BEV NMS.

**3. Model and rule.**
- **Candidate:** R2+T2+K40+G+RD.
- **Reference:** R2+T2+K40+G, seeds 1–3 (Phase 4).
- **The Phase 4 rule applies unchanged:**
  - The candidate must beat the reference on `select` by more than its seed SD at both 50–200 m and 100–200 m.
  - It must also pass the camera-removal veto.
- **Final, 6 seeds, scored once on `select` and val** whatever the rule says, because Phase 6 runs on the user's
  decision.
  - The rule decides only whether R2+T2+K40+G+RD replaces R2+T2+K40+G as the current model.
- **Radar gain:** against R0+T2+K40+G (no radar at all), seed against seed. The detector's own contribution is
  against R2+T2+K40+G.

**4. Also reported.**
- **SECOND ∪ radar:** SECOND's candidates plus the unmatched radar detections with their own scores, with no fusion
  network.
- **Recall gained:** scored cars beyond 50 m with a kept box (after NMS, final score ≥ 0.1) within 2 m, compared
  between models.
- **False alarms added:** kept radar-sourced boxes with final score ≥ 0.1 that are more than 2 m from every scored car,
  per frame.
- APCD by distance and condition against every comparison: SECOND, switch(R2+T2), R0+T2, R2+T2, R2+T2+K40,
  R0+T2+K40+G and R2+T2+K40+G.

## Phase 6b: radar-only candidates precise enough to help (written 2026-11-20, before any Phase 6b run)

**Scope** (the user's choice, 2026-11-20, after a literature search):
1. Precision filters on radar-only candidates.
2. Motion-corrected radar accumulation plus a per-point clutter classifier.
3. RadarDistill, using the authors' code.

Excluded: any camera-based confirmation (the camera is weak far away), and training on outside radar datasets (disk
and memory). Phase 6 showed why precision is the issue: radar-only candidates added recall but were 95–99% false.
Appended below every other box, their APCD gain is about their added recall times their precision.

**Detectors.**
- **D1:** the Phase 6 radar PointPillars (`runs/next_round/phase6/radar_pp`).
- **D2:** D1's config and schedule (40 epochs on `fit`) with two changes:
  - Doppler motion correction of older sweeps: each older return is moved along its line of sight by its Doppler
    speed times its age, as `Sequence._accumulate(motion=True)` does.
  - A seventh point feature: the car probability from a per-point clutter classifier.
- **Clutter classifier:**
  - A small MLP over the point's 6 features and its range.
  - Neighbourhood statistics within 2 m in BEV, over all accumulated sweeps: number of returns, number of distinct
    sweeps, mean and SD of Doppler speed, and maximum RCS.
  - Label 1 if the point lies inside a car box (any car label) enlarged by 0.5 m in length and width; 0 otherwise.
  - Trained on `fit` points for 5 epochs. Its AUC on `select` is reported.
- **D3: RadarDistill** (Bang et al., CVPR 2024; github.com/geonhobang/RadarDistill, commit a81d248), the authors'
  code for car only.
  - **Teacher:** a LiDAR PillarNet (their `pillarnet.yaml` network) trained on `fit` for 20 epochs, their teacher
    schedule.
  - **Student:** initialized from the teacher as their `ckpt.py` does, then distilled for 40 epochs, their schedule.
    Batch 4, adam_onecycle, learning rate 0.001, as in their configs.
  - **Range and pillars:** SECOND's x and y range. z is −1.5 to 2.5 m for LiDAR and −5 to 5 m for radar. Pillars are
    0.1 m, giving the same stride-8 BEV grid as SECOND (256 × 64).
  - **Augmentation:** their "distill" versions of flip (along x only, since the range is forward only), rotation
    ±5° and scaling 0.95–1.05, as for SECOND. No ground-truth sampling.
  - **Radar input:** the D1 or D2 point set, whichever has the higher detector-alone APCD at 50–200 m on `select`.
  - **The only changes to their code:**
    1. `ModulatedDeformConv` is replaced by `torchvision.ops.deform_conv2d` with the same arguments, because their CUDA
       op does not build on PyTorch 2.
    2. torch-scatter is installed for PyTorch 2.0 / CUDA 11.8.
    3. An aiMotive dataset class supplies LiDAR and radar points.
  - Their code is kept outside the repo at `~/RadarDistill`, with its ops built in place.
- Every detector exports boxes as Phase 6 does (BEV NMS 0.1, at most 100 per frame, score ≥ 0.01). Each is also scored
  on its own, by distance and condition, on `select`.

**Integration** (`clocs_r/radar_filters.py`, the same for every detector).
- **Radar-only candidates** as in Phase 6: detections with score ≥ 0.05 and no SECOND candidate of score ≥ 0.05 within
  2 m.
- **Filters:** score threshold t ∈ {0.05, 0.1, 0.2, 0.3, 0.5} × a cue, 25 filters in all. The cues:
  - none;
  - moving: at least one return with |v| ≥ 1 m/s in the gate (3 sweeps);
  - lidar: at least one LiDAR point in the box;
  - persist: a detection of the same detector in the previous frame, moved by the ego poses, within 2 m;
  - moving or persist.
- **Scoring modes:**
  - **last:** the final score is the radar score times a factor that puts every radar candidate below every kept box
    of the base, across the split. Appended at the end of the ranking, they cannot lower AP.
  - **calibrated:** the precision of that filter's candidates on `fit` in their distance bin (0–50, 50–100,
    100–200 m).
- **Bases:** R2+T2+K40+G (the current model, seeds 1–6) and SECOND alone. Radar candidates are merged with the base's
  scored boxes after the base model, without passing through the fusion network, then the usual BEV NMS.

**Rule** (on `select`).
1. A configuration is a detector, a filter and a mode.
   - It **beats** the base if its mean APCD gain over seeds 1–6, against R2+T2+K40+G, exceeds its own seed SD at both
     50–200 m and 100–200 m.
   - All 6 seeds are used because nothing is retrained per seed.
   - No camera-removal veto applies: radar candidates do not use the camera.
2. Among configurations that beat the base, the highest mean at 50–200 m wins and becomes the current model's radar
   add-on.
   - If none beats the base, the current model stays R2+T2+K40+G.
   - The configuration with the highest select mean at 50–200 m is still scored once on val, for the report.
3. **Final:** the chosen configuration on both bases, seeds 1–6, scored once on `select` and val. The report also
   gives:
   - precision and recall of every filter for every detector;
   - the clutter classifier's AUC;
   - each detector alone, by distance and condition;
   - every comparison: SECOND, switch(R2+T2), R0+T2, R2+T2, R2+T2+K40, R0+T2+K40+G and R2+T2+K40+G.

### Addendum to Phase 6b (2026-11-20, before any Phase 6b result): how radar candidates are merged

- Radar candidates are merged with the base model's boxes **after the base's own BEV NMS** (at most 100 per frame),
  and BEV NMS runs again on the union.
  - For "last" this is identical to merging before NMS, because NMS never lets a lower score suppress a higher one.
  - For "calibrated" this addendum fixes the procedure.
- "Last" scores are the radar score times half the lowest kept base score of the split.
- The selection table uses an APCD-only evaluator. It is tested to equal `metrics.evaluate`'s APCD exactly.
  - The reported finals use `metrics.evaluate` itself.
- D1's radar-only candidates are Phase 6's caches. D2 and D3 get their own, tagged with the detector name.

### Addendum to Phase 6b (2026-11-20, before any D3 run): RadarDistill details fixed while writing its configs

- **One range for both sensors.** Their framework uses a single point-cloud range for LiDAR and radar. Both use
  x 0–204.8 m, y ±25.6 m and z −5 to 5 m. The pillar encoders take z as a feature; the range only masks points.
- **Labels for teacher and student:** SECOND's rule, cars in the camera view with at least one LiDAR point. Their
  student is trained on the same boxes as the teacher, and distillation is meant to supply the evidence radar lacks.
- **Head:** their CenterHead without the velocity output (aiMotive labels have none), so 8 code weights.
- **Their losses, as published:** no range weighting.
- **Export:** CenterHead post-processing with score ≥ 0.01, NMS 0.2, at most 100 boxes, then our BEV NMS (0.1, at most
  100), as the other detectors are exported.
- **Code:** `clocs_r/radardistill_aimotive.py` and `clocs_r/configs/radardistill/{teacher,student,student_test}.yaml`.
- **torch-scatter** 2.1.2 (PyTorch 2.0, CUDA 11.8) was installed into the `clocs` environment for their pillar
  encoder.

### Addendum to Phase 6b (2026-11-20): closed after D1, by the user's re-evaluation (see REEVALUATION.md)

- **D1's audit is the Phase 6b result.** No filter of 25 and neither scoring mode gave a gain on `select`.
  - The best configuration was +0.00001 APCD at 50–200 m on top of R2+T2+K40+G.
  - Every filter's precision beyond 50 m was below about 3%.
- **Radar-created candidates are dropped from the framework.** In CLOCs only SECOND's candidates are rescored. The
  filter audit is therefore not run for D2 and D3, and val is not scored for Phase 6b.
- **D2 and RadarDistill (D3) are still trained, as planned in Phase 6b.** Their boxes are used in Phase 7 as a CLOCs
  sensor.
- This was decided before any D2 or D3 result existed.

## Phase 7: radar through features or through a detector inside CLOCs, and a situational selector (written 2026-11-20, before any Phase 7 run)

Context and the reasons for this structure: `REEVALUATION.md`. Fixed for every Phase 7 fusion network, whatever the
contribution ablations show:
- SECOND and Cascade R-CNN trained on `fit`;
- residual scoring T2, far weight 2, 3 epochs, seeds 1–6;
- radar gate δ0 0.5 m, κ 0.02, 3 sweeps;
- K40 for camera pairs.

**Experts.** All are CLOCs FusionNet models on SECOND's candidates, trained on `fit`.

| Name | Configuration | Inputs |
|---|---|---|
| L | SECOND alone | – |
| L0 | `R0+T2+NC` | no camera, no radar: the rescoring network's own effect (control) |
| LC | `R0+T2+K40` | camera; the Phase 3 weights, already trained |
| LRf | `R2+T2+NC` | radar as per-candidate Doppler features; no camera |
| LRd | `R0+T2+RP` | radar as a detector: radar boxes take the place of camera boxes in CLOCs' pairs; no camera, no radar point features |

- **NC:** the camera pairs are removed in training and at test; every candidate has only the reserved no-camera cell.
- **RP:** each SECOND candidate is paired with every radar detection (score ≥ 0.05) whose BEV center is within 2 m.
  - Each pair cell holds [affinity, radar score, s3D, distance], with affinity = 1 − d / 2 m. The affinity and radar
    score take the places of CLOCs' IoU and 2D score.
  - A candidate with no pair keeps the reserved cell.
  - The radar detector is the one of D1, D2 and D3 with the highest detector-alone APCD at 50–200 m on `select`.
- **Radar detectors:**
  - D2 is trained as in Phase 6b.
  - D3 (RadarDistill) uses a 20-epoch teacher and a 40-epoch student. If a smoke test projects more than 24 GPU hours
    for both, both schedules are halved (10 and 20 epochs). This rule is fixed now, before the timing is known.

**Selector S.**
- For each candidate, w = softmax(g(context)) weights the experts' logits {L, LC, LR\*}. The output is
  Σ w_k · logit_k, where logit_L = logit(s3D).
  - LR\* is whichever of LRf and LRd has the higher 50–200 m mean on `select`.
- **Context:**
  - distance;
  - the three expert logits;
  - camera evidence: the highest IoU × 2D score over K40 pairs, or −1;
  - radar evidence: log(1 + returns in the gate) and log(1 + moving returns);
  - LiDAR points and projected height (Phase 3 features).
- **g:** a small MLP (context → 16 → 3). Its last layer starts with zero weights and bias [2, 0, 0], so training
  starts close to LiDAR alone.
- **Training without stacking leakage:**
  - The selector is trained on `fit` with **out-of-fold** expert logits.
  - For each seed and each of `fit`'s three scene folds (`fold0`–`fold2`, Phase 1), LC and LR\* are trained on the other
    two folds and predict the held-out fold.
  - Then g is trained on all of `fit`: focal loss, far weight 2, 3 epochs, the same seed.
  - At test, the experts are the full-`fit` models of the same seed.
- The gate weights are plotted by distance and condition.

**Questions and comparisons** (APCD, primary at 50–200 m and 100–200 m, all bins and conditions reported):
- **RQ1, does radar help LiDAR beyond 50 m without the camera?** LRf − L0 and LRd − L0, seed against seed; each also
  against L.
- **RQ2, radar as features or as a detector?** LRd − LRf.
- **RQ3, does the selector help?**
  - S − max(L, LC, LR\*), where the best single expert is chosen on `select`.
  - S − L and S − R2+T2+K40+G (the incumbent).
- **Statistics:** paired seed differences (mean, SD, positive seeds) and scene-bootstrap 95% intervals (1,000
  resamples of scenes, means over seeds).

**Current-model rule** (on `select`, unchanged in kind):
- A Phase 7 model replaces the incumbent R2+T2+K40+G only if its mean beats the incumbent's by more than its own seed SD
  at both 50–200 m and 100–200 m.
- Among those that do, the highest 50–200 m mean wins.

**Val:** scored once, at the end, for every Phase 7 model and the earlier models together, after all are frozen. Val
was looked at before; see `REEVALUATION.md` section 5.

### Addendum to Phase 7 (2026-11-20, before any Phase 7 result): RadarDistill schedule, by the pre-registered 24-hour rule

- **Smoke tests,** each 200–300 iterations, measured while the contribution ablations were sharing the machine:

  | Stage | Speed | Iterations | Projected time |
  |---|---|---|---|
  | Teacher | 3.84 it/s | 89,980 | 6.5 h |
  | Student | 2.50 it/s | 179,960 | 20.0 h |
  | Total | | | **26.5 h** |

  The total is above 24 h, so by the rule both schedules are halved: **teacher 10 epochs, student 20 epochs.**
- **Settings of their own code that aiMotive needs** (no change to their code beyond the deformable-convolution swap
  already recorded):
  - their head option `waymo: True`, which takes boxes without velocity (7 values plus class);
  - empty stand-ins for dataset devkits that their package imports at load time but aiMotive never uses
    (nuscenes, kornia, av2, waymo_open_dataset, and an unused openpyxl import in their backbone);
  - at export, the student's weights (`radar_*`) and the step counter are loaded, and the teacher's are not.
- **Model sizes:** teacher 14.8M parameters; student 23.5M trainable, with the frozen teacher alongside (38.3M in
  all).

### Addendum to Phase 7 (2026-11-21, before any student result): a second change to RadarDistill's code

- Their `Detector3DTemplate.build_radar_vfe` passes a constant 6 as the radar feature count, which is nuScenes radar's
  6 features.
- The pre-registered rule chose D2's radar input, which has 7: D1's 6 plus the clutter probability. The student
  crashed at its first step.
- The constant is replaced by the count the dataset reports, which is 6 for 6-feature inputs, so their behaviour is
  unchanged there.
- The teacher, which has no radar branch, is unaffected and was kept.

### Addendum to Phase 7 (2026-11-21, after the `select` results, before val is looked at): post-hoc fixed-blend ablation

**Why, and why it is post hoc:**
- On `select` the selector's gate weights barely vary with distance or condition (L 0.01–0.10, LC 0.60–0.70,
  LRd 0.24–0.35).
- So S's gain over the single experts may come from averaging experts, not from choosing by situation.
- This ablation was added **after** the Phase 7 `select` results were seen, and is labelled post hoc everywhere it is
  reported.

**The model, B(R0+T2+RP):**
- The same as S, except that the context MLP g is replaced by three constants: w = softmax(b), with b ∈ ℝ³ trained.
- b starts at [2, 0, 0], as S's bias does.
- Everything else is identical:
  - the out-of-fold expert logits on `fit` (the same three scene folds and seeds 1–6);
  - the loss, far weight, epochs, optimiser and schedule;
  - the test-time experts.
- So the one difference between S and B is whether the weights can depend on the candidate's situation.

**Reported, on `select` and on val:**
- S − B, the main comparison.
- B − the best single expert on `select`, B − L, and B − the incumbent.
- All bins and conditions, seed-paired differences, scene-bootstrap 95% intervals, and B's learned weights.

**Interpretation, fixed now:**
- The situational gate "adds beyond averaging" only if S beats B by the Phase 7 rule on `select`: S's mean is higher by
  more than S's seed SD at both 50–200 m and 100–200 m. Val is reported alongside.
- Otherwise the report says that S's gain is explained by a fixed blend of the experts.
- **B cannot become the current model under the Phase 7 rule,** because it was defined after the `select` results were
  seen. If B matches S, the report names it as the simpler form, to be confirmed on data not yet used.

**Order:** B is trained and scored on `select`, then val is scored once for every Phase 7 model, B included.

### Addendum to Phase 7 (2026-11-21, after val was scored): oracle ceiling of situation-dependent blends

**Why:**
- The fixed blend B matched or beat the selector S on `select` and on val.
- Before any further selector work, this measures how much a perfect choice of blend weights by situation could add
  over B at all.
- It is a diagnostic in the manner of Phase 5. It is an upper bound tuned on `select` itself, so it is never a model
  result and is never scored on val.

**Method** (`clocs_r/selector_ceiling.py`, each of seeds 1–6):
- The expert logits on `select` are the full-`fit` experts of that seed, as at test time: L, LC = R0+T2+K40,
  LRd = R0+T2+RP with D2's boxes.
- The candidates are scored as sigmoid(Σ w_k z_k) for every w on the simplex in steps of 0.05 (231 weightings).
- The NMS and APCD are exactly those of `metrics.evaluate`; a test checks the two are equal.
- **Global oracle:** the best single w per bin.
- **Distance oracle:** the best w for each distance bin (0–30, 30–50, 50–100, 100–200 m). Each candidate is then
  scored with its own bin's w, and the result is evaluated once.
- **Distance × condition oracle,** secondary: the same, per condition label (highway, urban, night, rain). The labels
  are annotations not available at run time, so this is an upper bound only.
- **Check:** B's and S's APCD recomputed by the tool must agree with `blend.json` and `selector.json` to within 0.0005.

**Decision rule, fixed now:**
- Further selector variants are pre-registered and run only if the distance oracle beats B by at least 0.005 at
  100–200 m (mean over seeds).
- Otherwise selector work stops, and the report states the ceiling.
- All oracles are reported in every bin, together with the distance oracle minus the global oracle (the part due to
  distance dependence rather than to tuning on `select`).
- If the rule allows the variants, there are two: S with radar-detector evidence in the context, and the selector
  over the three detectors' own scores (SECOND, Cascade R-CNN, the radar detector).

### Addendum to Phase 7 (2026-11-21, after val was scored): one CLOCs network over all three detectors (LCR)

**Why** (the user's decision): a final model that stays within CLOCs' structure and is as simple as possible. That
means one fusion network, instead of two networks combined by a selector or a blend.

**This is post-val development** and is labelled as such everywhere it is reported. Val was scored for Phase 7 on
2026-11-21. This model is scored on val once more, alone, after it is frozen.

**The model, LCR = `R0+T2+K40+RC`** (token RC, `clocs_r/radar_pairs.py`):
- **Candidates:** SECOND's 3D candidates only. Neither radar nor the camera creates candidates.
- **One CLOCs FusionNet** (the same four 1×1 convolutions, the T2 residual on SECOND's logit, and the max over a
  candidate's cells) reads two kinds of pair cells per candidate:
  - **camera cells** (Cascade R-CNN's boxes with the K40 floor, exactly as in LC):
    [IoU, 2D score, s3D, distance, MISSING, MISSING];
  - **radar cells** (D2's boxes, exactly as in LRd: score ≥ 0.05, centre within 2 m, affinity = 1 − d/2 m):
    [MISSING, MISSING, s3D, distance, affinity, radar score];
  - a candidate with neither keeps one reserved cell [MISSING, MISSING, s3D, distance, MISSING, MISSING].
- So per candidate, the network fuses SECOND with the camera, with the radar detector, with both, or with neither,
  according to which detectors produced matching boxes. It learns how much to trust each combination.
- **Each sensor enters once:**
  - radar only through D2's boxes, with no radar point features (R0);
  - the camera only through its boxes.
- **Training identical to LC and LRd:** `fit`, 3 epochs, focal loss, far weight 2, gate (0.5, 0.02), seeds 1–6.

**Comparisons** (APCD, all bins and conditions; seed-paired differences; scene-bootstrap 95% intervals on `select` and
val):
- **LCR − LC:** do radar-detector cells add to camera CLOCs? This is the one-network form of the main question.
- **LCR − LRd:** do camera cells add to radar CLOCs?
- **LCR − B:** one network against the two-network blend.
- **LCR − L:** against SECOND alone.

**Decision rule, fixed now** (on `select`):
- LCR becomes the final model of the CLOCs version if its mean is not below B's by more than LCR's own seed SD, at
  both 50–200 m and 100–200 m.
- Otherwise B remains the simplest adequate form.
- Val is reported for both whatever the rule says. No setting is changed after val.

**Note added after the run (2026-11-21, about 22:05 UTC):**
- The rule chose B on `select` (`phase7/decision_lcr.json`).
- The user then chose LCR as the main model of the CLOCs version, for simplicity. This is a recorded departure from the
  rule above. See `DECISIONS.md` items 12–13.
- B, LC and LRd remain in every table as comparisons.

### Addendum to Phase 7 (2026-11-21, after LCR's `select` and val results): LCR-D, camera dropout beyond 50 m in training

Written at 22:25 UTC, before any LCR-D run. This is post-val development, and it is **val-informed**:
- The idea (make the one network rely on radar more beyond 50 m) came after seeing that, on val, LCR follows the camera
  and loses to SECOND beyond 50 m, while the radar network LRd does not.
- A val gain for LCR-D is therefore not independent confirmation, and is reported as such.
- `select` decides; nothing is tuned on val.

**Why:**
- LCR's radar-detector cells add almost nothing over camera CLOCs (LCR − LC: +0.001 on `select` and +0.0007 on val at
  50–200 m).
- The leading explanation (`DECISIONS.md` item 12) is that on `fit`, the camera's far boxes look far more reliable than
  on unseen scenes, so the network learns to lean on them and never learns radar.
- Hiding the camera for far candidates during training forces the network to learn what radar evidence is worth. At
  test time it sees everything and can use either sensor.

**Model:** `R0+T2+K40+RC+DF`, which is LCR plus token DF (`fusion.drop_far_camera`, `tune_fusion.DF_M`, `DF_P`).
- In training only, at every step, each SECOND candidate beyond 50 m loses its camera cells with probability 0.5.
  - Its radar-detector cells and its own score and distance stay.
  - A candidate left with no cell gets the reserved cell.
  - The draws come from the separate dropout stream (seed + 10,000), so the frame order is LCR's.
- Test time is identical to LCR.
- 50 m and 0.5 are fixed now and not varied:
  - 50 m is the project's far boundary;
  - 0.5 means the network trains on both situations equally.
- **Training otherwise identical to LCR:** `fit`, 3 epochs, focal loss, far weight 2, gate (0.5, 0.02), seeds 1–6.

**Comparisons** (APCD, all bins and conditions; seed-paired differences; scene-bootstrap 95% intervals on `select` and
val):
- **LCR-D − LCR:** the effect of the dropout. This is the main question.
- **LCR-D − LC:** does the one network now use radar?
- **LCR-D − LRd:** against radar CLOCs.
- **LCR-D − L:** against SECOND alone.

**Decision rule, fixed now** (on `select`):
- LCR-D replaces LCR as the main model if its mean is not below LCR's by more than LCR-D's own seed SD in each of
  0–30, 30–50, 50–200 and 100–200 m. It must not lose up close to gain far away.
- Otherwise LCR stays.
- Val is scored once for LCR-D whatever the rule says. No setting is changed after val.
