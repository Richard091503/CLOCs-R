# Fusion networks trained on `select`: predictions and fixed settings

Written and committed on 2026-11-26, before any network here was trained and before any val result of these models
existed. Nothing below is changed after training starts. Every result is reported, whichever way it comes out.

## Question

The paper explains why CLOCs-R ignores radar and trails SECOND beyond 50 m on val as follows. The fusion network was
trained on `fit`, the scenes Cascade R-CNN was trained on, so it learned to trust the camera there. That trust does not
carry over to new scenes: Cascade R-CNN's recall at 100–200 m is 0.911 on `fit`, 0.353 on `select` and 0.227 on val.
No detector saw `select` (24 scenes, 11 recordings). Training the fusion networks on `select` therefore tests the
explanation directly.

## What changes, and what does not

The only change is the fusion networks' training frames: `select` (3,409 frames) instead of `fit` (17,993).

Unchanged:
- the frozen detectors and the cached cells (`runs/lcr`, the same files as the paper's runs);
- the K40 floor, the 2 m radar pairing radius and the reserved cell;
- the residual score (Eq. 3) and the focal loss (α 0.25, γ 2);
- range weighting (far weight 2);
- AdamW with the one-cycle schedule, and one frame per step;
- NMS, the metric and the bootstrap.

| Paper name | Code | Cells |
|---|---|---|
| L0_sel | `R0+T2+NC` | the reserved cell only (sensor-free control) |
| LRd_sel | `R0+T2+RP` | D2's boxes only |
| LC_sel | `R0+T2+K40` | Cascade R-CNN cells only |
| CLOCs-R_sel | `R0+T2+K40+RC` | camera and radar cells |
| B_sel | blend of SECOND, LC_sel and LRd_sel logits | weights learned on `select` |

Each is compared with its fit-trained counterpart, scored from the paper's saved networks:
- L0: `runs/next_round/phase7/fusion`;
- LRd: `phase7/fusion_rp`;
- LC: `phase3/fusion`;
- CLOCs-R: `phase7/fusion_lcr`;
- B: thesis `results/next_round/phase7/blend_R0+T2+RP_s*.pth`, with that seed's LC and LRd.

**Code.**
- `clocs_r/tune_fusion.py` gains two options, whose defaults leave every existing run unchanged:
  - token **NC**: every candidate gets only the reserved cell `[-1, -1, s3D, d]`. These are the thesis branch's NC cells
    and settings, and its saved L0 networks load unchanged.
  - **`--steps`**: the total optimizer steps.
- `clocs_r/train_fusion.fit` gains `steps=None`. When it is given, training draws fresh frame permutations as long as
  needed and cuts the last one off at `steps`. The one-cycle schedule spans exactly `steps`.
- B's blend is ported from the thesis branch (`selector.py --blend`) to `results/select_trained/blend.py`.
- `check_equivalence.py` trains each kind of network with both code bases on the same 200 `select` frames (2 epochs,
  seed 1) and requires identical weights. It covers LC, L0 (NC), LRd, CLOCs-R and the blend, plus a `--steps` run that
  must stop after exactly the requested number of updates. **If any weight differs, the run stops.**

## Training length (fixed)

- **Primary: 53,979 optimizer steps** per network, as every fit-trained fusion network (3 epochs × 17,993 `fit`
  frames). On `select` that is 15.83 passes: 15 full permutations plus 2,844 steps of a 16th.
- **Secondary, labeled as such: 3 epochs over `select`**, 10,227 steps.
- **Seeds 1–6** for every model; 3 seeds train at a time on the GPU, as in the paper's runs.

## B_sel (fixed)

B is trained as it was on `fit`, with `select` in its place:
1. **Folds.** The 3 folds of whole recordings use the same rule as `fit`'s (`splits._fold_of`): per condition,
   recordings from the largest down, each to the fold with the fewest frames of that condition so far, ties to the
   lower index. Applied to `select`'s recordings, this gives:

   | Fold | Recordings | Frames |
   |---|---|---|
   | 0 | 20220202-105828@Cali (highway), 20210701-225716@Yoda (night), 20210427-130007@Jarvis (rain), 20210729-162718@Yoda (urban) | 1,845 |
   | 1 | 20210401-120439@Jarvis (highway), 20210729-223722@Yoda (night), 20210625-175142@Yoda (urban) | 994 |
   | 2 | 20220204-100251@Cali (highway), 20210817-214850@Yoda (night), 20210724-064134@Yoda, 20210630-130826@Yoda (urban) | 570 |

   The folds are uneven because `select` has 1–4 recordings per condition. They are kept as the rule makes them, not
   rebalanced.
2. **Out-of-fold experts.** For each fold, LC and LRd are retrained on the other two folds (same seed) and score the
   held-out fold.
   - On `fit` the out-of-fold experts took the same number of passes as the full experts (3 epochs).
   - Here they also take the full experts' 15.83 passes, i.e. round(53,979 × frames / 3,409) steps: 24,765, 38,240
     and 44,953 steps on 1,564, 2,415 and 2,839 frames.
   - Secondary: 3 epochs (4,692, 7,245 and 8,517 steps).
3. **Blend.** w = softmax(b), b started at (2, 0, 0), trained on the out-of-fold logits of all `select` frames with
   the thesis loss and schedule (focal loss with range weights, AdamW, one-cycle). Primary 53,979 steps, secondary
   10,227.
4. **At test,** the experts are LC_sel and LRd_sel of the same seed and length.

## Order of work and val exposure

1. **Step 0.** Retrain fit-trained CLOCs-R seed 1 with this code, and score it on val. Its 50–200 m APCD must equal
   the stored 0.4256 (`results/lcr/data/lcr_val.json`) to 3 decimals, or **the run stops** and nothing else is trained.
2. The equivalence check (above).
3. All training: primary, then secondary. No val scoring of any select-trained model happens during training. Each
   network is scored on `select` only (in-sample, recorded, not used for anything).
4. Val, **once per model**, after every run has finished (`score_val.py`). The fit-trained models are scored from
   their saved networks in the same pass, and must reproduce their stored per-seed val results.

## Evaluation (fixed)

- **Metric.** The paper's APCD:
  - centre-distance thresholds 0.5, 1, 2 and 4 m;
  - cars 0–204.8 m ahead, ±25.6 m to the side, in the front camera's view;
  - BEV NMS at 0.1.
- **Per model:** val APCD in all, 0–30, 30–50, 50–100, 100–200 and 50–200 m, and at 50–200 m by condition (highway,
  urban, night, rain). Mean of seeds 1–6 and seed SD (ddof 1).
- **Bootstrap.** The paper's code:
  - per-frame matches from `clocs_r/bootstrap.py`;
  - 1,000 resamples, `RandomState(0)` per level, seeds averaged inside each resample, percentile 95% intervals.
  - Two levels: **scene** (the verdicts use this level) and **recording** (`results/review_round/cluster_bootstrap.py`).
    Val has 25 scenes from 8 recordings, so the recording intervals are themselves rough.
  - The point estimate is the difference of the 6-seed means, seed-paired (seed k against seed k).
- **Differences at 50–200 and 100–200 m** (50–100 m also computed):
  - each select-trained model minus its fit-trained counterpart;
  - CLOCs-R_sel − LC_sel and LRd_sel − L0_sel (radar);
  - LC_sel − L0_sel and CLOCs-R_sel − LRd_sel (camera);
  - each select-trained model minus SECOND;
  - B_sel − CLOCs-R_sel;
  - secondary: CLOCs-R_sel3 − LC_sel3, CLOCs-R_sel3 − CLOCs-R and CLOCs-R_sel3 − SECOND.
- **Winning cell** (`results/qualitative/`'s definition):
  - Every val car at 50–200 m whose SECOND candidate is within 2 m counts once. Candidates are matched greedily in
    descending SECOND score, among the scored candidates.
  - The cell that wins CLOCs-R's maximum at that candidate is counted as camera, radar or reserved, in 50–100 and
    100–200 m by the car's distance.
  - The candidates are the same for every network, since the matching uses SECOND's scores.
  - Reported per seed and as the mean of the 6 per-seed shares, for fit-trained CLOCs-R and CLOCs-R_sel. Fit-trained
    seed 1 must reproduce `results/qualitative/` exactly (82.9% and 58.8% camera).
  - Scene and recording bootstraps of the share difference: per resample, each seed's share from the weighted counts,
    then the mean of seeds.

## Hypotheses and decision rules (val, 6 seeds, scene-level 95% interval)

Each is **supported**, **not supported** or **contradicted**:

**H1 (radar gets used once the camera isn't over-trusted).** Δ = CLOCs-R_sel − LC_sel at 50–200 m. Fit-trained:
+0.001 [−0.004, +0.005].
- Supported: Δ ≥ +0.005 and the interval's lower end > 0.
- Contradicted: the upper end < 0.
- Otherwise not supported, including an interval above 0 with Δ < +0.005.

**H2 (less camera reliance).** Δ_b = camera share of CLOCs-R_sel − camera share of fit-trained CLOCs-R (6-seed means),
in b = 50–100 m and b = 100–200 m. Fit-trained seed 1: 82.9% and 58.8%.
- Supported: Δ_b < 0 with the upper end < 0, in both bins.
- Contradicted: Δ_b > 0 with the lower end > 0, in either bin.
- Otherwise not supported.

**H3 (the gap to LiDAR shrinks).** CLOCs-R_sel − SECOND at 50–200 m against the fit-trained −0.018. SECOND is the same
model in both differences, so the test is Δ = CLOCs-R_sel − CLOCs-R (fit-trained), seed-paired.
- Supported: lower end > 0.
- Contradicted: upper end < 0.
- Otherwise not supported.
- Also reported: CLOCs-R_sel − SECOND with its interval, and whether that interval excludes zero, in either direction.

**H4 (the rescoring loss comes from fit's calibration).** L0_sel − SECOND at 50–200 m against the fit-trained −0.017
(6-seed mean 0.4282 against 0.4455). Δ = L0_sel − L0 (fit-trained).
- Supported: |L0_sel − SECOND| < |L0 − SECOND| and the lower end of Δ's interval > 0.
- Contradicted: Δ's upper end < 0, i.e. further from zero, on the negative side.
- Otherwise not supported. A result above SECOND by more than 0.017 is also "not supported" under this wording; it
  would be reported as such, with its sign.

The secondary 3-epoch runs get H1 and H3 by the same rules, labeled secondary. The verdicts that count are the
primary ones. Numbers are taken from the result files and rounded to 3 decimals in the report.

## Outputs

`results/select_trained/`:
- `PREDICTIONS.md` (this file);
- `step0.json`, `equivalence.json`;
- `val.json` (every val number, from `score_val.py`);
- `results.json`, `REPORT.md`.

Training logs and networks go to `runs/select_trained/`. Nothing else in `results/` is changed.
