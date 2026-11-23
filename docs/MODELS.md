# Models: structure, data and how to rebuild them

This branch (`master`) runs the LCR version and its comparison models end to end (`clocs_r/run_lcr.sh`). Everything else
built during the project is described below, in enough detail to rebuild it. Its code, run scripts, logs and results are
on the **`thesis`** branch, unchanged:

```bash
git fetch origin thesis && git worktree add ../CLOCs-R-thesis thesis   # or: git checkout thesis
```

On that branch, `results/next_round/` holds:
- `PLAN.md`: every pre-registered rule, written before each run;
- `DECISIONS.md`: dated decisions and findings;
- `README.md` and `phase*/report.md`: the results;
- `phase*_run.sh`: the exact commands.

`results/full_dataset/` and `results/clocs_radar/` hold the earlier CLOCs-R work.

## Shared setup

**Data.** aiMotive multimodal dataset, all four conditions (highway, urban, night, rain).
- `clocs_r/fetch_aimotive.py` keeps the front camera, the LiDAR points in its view, the front radar, the labels and the
  ego poses.
- Cars are CAR, VAN and PICKUP; TRUCK, BUS and TRAILER are ignored in evaluation.
- Region: x 0–204.8 m, |y| ≤ 25.6 m.

**Splits** (`clocs_r/splits.py`):

| Split | Scenes | Frames | Role |
|---|---|---|---|
| `fit` | 127 | 17,993 | trains every detector and every fusion network |
| `select` | 24 (11 whole recordings) | 3,409 | chooses every setting; no recording is also in `fit` |
| val | 25 | 5,181 | aiMotive's validation set, scored once per model after selection |

`select` is about 15% of each condition's training frames and 20% of its cars at 100–200 m. Val was scored during
development (the exposure log is in the thesis record), so it is a development set, not an untouched test set.

**Metric.** APCD: average precision with greedy centre-distance matching in BEV, averaged over the 0.5, 1, 2 and 4 m
thresholds (`clocs_r/metrics.py`).
- Reported by distance bin: 0–30, 30–50, 50–100 and 100–200 m, plus 50–200 m and all.
- Boxes pass BEV NMS at 0.1 first.
- Fusion models: 6 seeds; mean ± SD, seed-paired differences, and scene-bootstrap 95% intervals (`clocs_r/bootstrap.py`,
  1,000 scene resamples).

**Detectors** (all trained on `fit`):

| Detector | Structure | Training | Code |
|---|---|---|---|
| SECOND | OpenPCDet SECOND with a range-weighted anchor head (`second_head.py`) | 40 epochs | `train_second.py`, `configs/second_aimotive.yaml` |
| Cascade R-CNN | MMDetection 2.28 R50-FPN, images upscaled 1.5× | 4 epochs, batch 2 × 2 accumulation | `cascade_2d.py` |
| D2 (radar) | OpenPCDet PointPillars on 5 radar sweeps with Doppler motion correction; 7 point features (x, y, z, Doppler, RCS, age, clutter probability) | 40 epochs | `radar_dataset.py`, `radar_clutter.py`, `configs/pointpillar_radar_d2_aimotive.yaml` |

The clutter classifier is a small MLP over each radar return's features and its 2 m neighbourhood. Its label is "inside
a car box enlarged by 0.5 m", and it is trained on `fit`.

## The CLOCs fusion network (all CLOCs models below)

CLOCs (Pang et al., IROS 2020) rescores a LiDAR detector's candidates using another detector's boxes. It does not
create candidates.
- **Cells.** For each SECOND candidate j and each paired detection i there is one cell
  `[IoU_ij, s_i, s3D_j, d_j]`:
  - IoU between the candidate's image projection and the 2D box;
  - the 2D score;
  - SECOND's score;
  - the candidate's range divided by 204.8 m.
  A candidate with no pair gets the reserved cell `[-1, -1, s3D_j, d_j]`.
- **Network.** Four 1×1 convolutions, 4 → 18 → 36 → 36 → 1 (`fusion.FusionNet`). The output is the maximum over each
  candidate's cells.
- **T2 (residual).** The output is added to logit(s3D). The last layer starts at zero, so training starts exactly from
  SECOND's ranking.
- **Training.** Focal loss (α 0.25, γ 2). Targets: 1 above BEV IoU 0.7 with a car, 0 below 0.5, ignored in between.
  - Range-weighted loss (`range_weight.py`, far weight 2): candidates beyond 60 m count double, ramping up from 40 m.
  - AdamW with a one-cycle schedule, 3 epochs on `fit`.
- **Inference.** Sigmoid, then BEV NMS at 0.1.

**Configurations are named by tokens** (`clocs_r/tune_fusion.py`).
- On this branch: R0, T2, K40, RP, RC and DF.
- On the thesis branch: also R1–R7, T1, T3–T5, H, P, I1, I2, D2, D3, S50, S100, G, C, RD and NC.

## Models on this branch (executable)

| Label | Configuration | What the second sensor contributes |
|---|---|---|
| L | SECOND alone | nothing |
| LC | R0+T2+K40 | Cascade R-CNN boxes as pairs. **K40:** each candidate keeps only 2D boxes scoring at least the frame's 40th-highest 2D score (`visibility.TOP_2D`, `fusion.filter_pairs`). |
| LRd | R0+T2+RP | D2's boxes as the pairs, in the camera's places. Every D2 box with score ≥ 0.05 within 2 m (BEV centre distance d) of the candidate gives the cell `[1 − d/2, radar score, s3D, d_j]` (`radar_pairs.py`). |
| LCR | R0+T2+K40+RC | **One network over all three detectors.** Camera cells `[IoU, s2D, s3D, d, −1, −1]` and radar cells `[−1, −1, s3D, d, affinity, radar score]` side by side, plus the reserved cell `[−1, −1, s3D, d, −1, −1]` for a candidate with neither (`radar_pairs.combined_cells`). 6 input channels. |
| LCR-D | R0+T2+K40+RC+DF | LCR, except that in training only, each candidate beyond 50 m loses its camera cells with probability 0.5 at every step (`fusion.drop_far_camera`). Test time is identical to LCR. |

## Models on the thesis branch (documented here)

Each entry gives the structure, where it is on the `thesis` branch, and what it found.

### Earlier CLOCs-R: radar as point features in the cells (before the next round)

- **Structure.** The CLOCs cell gains m radar channels per candidate, from the radar returns in a gate around it.
  - **Gate (Eq. 2):** a return counts for candidate j if it lies inside the box enlarged on every side by
    δ = δ0 + κ·ρ_j (in the box's own BEV frame), with δ0 0.5 m, κ 0.02 and ρ_j the candidate's range.
  - **Features** (`radar_features.py`), over the gated returns:
    - r = log(1 + returns);
    - v = median |ego-compensated Doppler| / 40 m/s;
    - e = median absolute deviation of the Doppler / 40 m/s;
    - δ = |median return range − ρ_j| / gate size;
    - o = distance from the box centre to the nearest return / gate size.
    With no return, r = 0 and the others are −1.
  - **Feature sets:** R0 (none), R1 (r), R2 (v), R3 (r, v), R4 (r, v, e), R5 (r, δ, o), R6 (r, v, δ), R7 (all five).
  - **Other tokens:** T1 (far targets: IoU 0.5/0.3 beyond 50 m), T3 (3D score as a scaled logit), T4a/T4b (5 or 10
    sweeps with motion correction), T5 (RCS channel).
- **Code:** `cache_cells.py` (with radar), `train_fusion.py` and `tune_fusion.py` on the thesis branch, `ablation_full.py`,
  `run_full.sh`.
- **Results:** `results/full_dataset/`, `results/clocs_radar/`.
- **Phase 0 reproduction** (val, 50–200 m, 6 seeds):
  - R0+T2 (CLOCs): 0.4208 ± 0.0070;
  - R2+T2 (CLOCs-R): 0.4199 ± 0.0097;
  - SECOND: 0.457.

### Range switch (Phase 2)

- **Structure.** A trained CLOCs network's score within 50 m, and SECOND's own score from 50 m on. No training.
- **Code:** `range_switch.py`.
- **Finding:** beyond 50 m it is SECOND by construction, so it cannot answer whether radar helps there.

### Camera-visibility and pairing variants (Phase 3)

- **Tokens on R2+T2:**
  - H: projected 2D height;
  - P: log LiDAR points in the box (both `visibility.py`, appended per candidate);
  - I1, I2: pair IoU > 0.1 or 0.25;
  - K40;
  - D2, D3: whole-frame camera dropout of 0.2 or 0.3;
  - S50, S100: oversampling frames with far cars.
- **Finding:** only K40 passed its rule, with +0.0043 at 50–200 m and +0.0186 at 100–200 m on `select`. K40 is kept on
  this branch.

### Gated fusion (Phase 4; "inc", R2+T2+K40+G)

- **Structure** (`fusion.GatedFusion`): the score is logit(s3D) + g_cam · camera term + g_radar · radar term.
  - The camera term is the maximum over cells of a small MLP.
  - The radar term is a per-candidate MLP over the radar features.
  - The gates are sigmoids of projected height, LiDAR points, radar returns, moving returns and distance; C adds a
    one-hot driving condition.
  - Both terms start at zero.
- **Size:** 2,626 parameters.
- **Finding:** adopted on `select`, but on val it scored 0.4199 (50–200 m), below SECOND's 0.4455.

### Oracle injection (Phase 5)

- **Structure.** The ceiling of a radar detector: each far car with no LiDAR point and a radar return gets its
  ground-truth box added as a candidate (s3D 0.1). Each model then rescores it ("low"), or its score is set to 1
  ("perfect").
- **Code:** `oracle_inject.py`.
- **Targets:** 4.0% of far cars on `select`, 5.3% on val.

### Radar detector boxes as new candidates (Phase 6, token RD) and precision filters (Phase 6b)

- **RD:** radar detections (score ≥ 0.05) with no SECOND candidate of score ≥ 0.05 within 2 m become candidates. Each
  SECOND candidate gets the highest radar score within 2 m as a channel (`radar_candidates.py`). Its `build` remains on
  this branch, because D2's own score reuses that cache.
  - **Finding:** the new candidates added recall but were 95–99% false.
- **Phase 6b filters** (`radar_filters.py`): score threshold plus a cue (moving return, LiDAR point, persistence across
  frames), merged after NMS.
- **Phase 6b detector:** D2, the detector kept here.

### Radar detectors D1 and D3

| Detector | Structure | `select` APCD, 50–200 m |
|---|---|---|
| D1 | the same PointPillars as D2, without motion correction or clutter probability (`configs/pointpillar_radar_aimotive.yaml`) | 0.199 |
| D2 | kept | 0.217 |
| D3 | RadarDistill (Bang et al., CVPR 2024; the authors' code): LiDAR PillarNet teacher 10 epochs, radar student 20 epochs, schedules halved by a pre-registered time rule (`radardistill_aimotive.py`, `configs/radardistill/`) | 0.215 |

### Radar as features without a camera (Phase 7 controls)

- **L0 (R0+T2+NC):** CLOCs with neither camera nor radar.
- **LRf (R2+T2+NC):** radar point features, no camera.
- **Val results** (50–200 m / 100–200 m):
  - L0: 0.4282 / 0.2729;
  - LRf: 0.4233 / 0.2731;
  - LRd: 0.4415 / 0.2858.
- **Bootstrap:** LRd − L0 is +0.0133 [+0.0049, +0.0228] at 50–200 m on val. Radar through D2's boxes helps; radar as
  point features does not.

### Selector S and fixed blend B (Phase 7)

- **S** (`selector.Gate`): per candidate, w = softmax(g(context)), with output Σ w_k · logit_k over the experts
  {L, LC, LRd}.
  - g is an MLP 10 → 16 → 3. Its context is distance, the three logits, camera evidence, radar returns, moving returns,
    LiDAR points, projected height and s3D.
  - It starts close to L.
  - It is trained on `fit` with out-of-fold expert logits (3 folds).
- **B** (`selector.Blend`): one softmax weight vector for every candidate.
  - Learned weights: L 0.00, LC 0.72, LRd 0.28.
  - B ≥ S on `select`.
  - **B was the pre-registered final model**: val 0.4428 / 0.2912 against SECOND's 0.4455 / 0.2841.
- **Ceiling** (`selector_ceiling.py`): the best blend per distance bin, chosen with labels, beats B by only +0.0025 at
  100–200 m. So no situational selector was pursued.

### Cross-fitting (Phase 1)

- **Structure.** SECOND and Cascade R-CNN were retrained on each 2/3 of `fit` (folds by recording, `splits.FOLDS`). The
  fusion networks then trained on out-of-fold candidates.
- **Finding:** the rule kept in-sample training. With cross-fitted inputs, though, radar features gained +0.009 on
  `select` (3/3 seeds), against 0.000 in-sample. Cross-fitted inputs for LCR have not been tried. It is the leading
  explanation for LCR under-using radar.

### KITTI reproduction of the original CLOCs

- **Code:** `second/` and `torchplus/` (the original CLOCs code), `eval_kitti.py`, `kitti_check.py`.
- **Results:** `results/clocs_no_radar/kitti_*`. Upstream: <https://github.com/pangsu0613/CLOCs>.
