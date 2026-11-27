# Size control for the select-trained result: predictions and fixed settings

Written and committed on 2026-11-27, before any network was trained on `fit_sub` and before any val result of these
models existed. Nothing below changes after training starts. Every result is reported, whichever way it comes out.

## Question

Fusion networks trained on `select` (results/select_trained) used radar and leaned less on the camera than the
networks trained on `fit`. The `select` set has 3,409 frames in scenes no detector saw; `fit` has 17,993 frames in the
detectors' own training scenes. The two runs therefore differ in two ways:
- whether the detectors saw the training scenes;
- the size of the training set.

This control separates the two. It trains the same networks on `fit_sub`, a subset of `fit` scenes the size of
`select`.
- If the camera reliance comes back, the effect is about the detectors having seen the scenes.
- If it doesn't, the effect is about the size of the training set.

## What changes, and what does not

The training frames are the only change from the select-trained primary runs: `fit_sub` (3,397 frames) instead of
`select`.

Unchanged from those runs:
- the frozen detectors and the cached cells (`runs/lcr`);
- the K40 floor, the 2 m radar pairing radius and the reserved cell;
- the residual score (Eq. 3) and the focal loss (α 0.25, γ 2);
- range weighting (far weight 2);
- AdamW with the one-cycle schedule, and one frame per step;
- **53,979 optimizer steps**, the select-trained primary runs' `--steps`. On `fit_sub` that is 15.89 passes: 15 full
  permutations plus 3,024 steps of a 16th;
- seeds 1–6, three at a time on the GPU;
- NMS, the metric and the bootstrap.

| Name | Code | Cells |
|---|---|---|
| L0_sub | `R0+T2+NC` | the reserved cell only (sensor-free control) |
| LRd_sub | `R0+T2+RP` | D2's boxes only |
| LC_sub | `R0+T2+K40` | Cascade R-CNN cells only |
| CLOCs-R_sub | `R0+T2+K40+RC` | camera and radar cells |

There is no B_sub; the prompt does not ask for one.

**Code.** `clocs_r/tune_fusion.py` gains `--train_scenes FILE`. Its default leaves every existing run unchanged.
- It keeps only the frames of the listed scenes of `--train_split` (here `fit`), in the cached order.
- The result is the frame list a separate cache of those scenes would hold, so no new cache is written.
- `fit_base.pkl` is 4.9 GB, and the disk has 4.6 GB free.
- The run checks that every network trained on exactly the subset's 3,397 frames.

## The subset `fit_sub` (fixed; `subset.json`, from `draw_subset.py`)

**Rule.**
1. For each driving condition, take `fit`'s scenes in `clocs_r.splits` order and shuffle them with a fresh
   `np.random.RandomState(0)`.
2. Take scenes in that order. Stop at the first prefix whose kept-frame count is closest to `select`'s count for the
   condition. Frame counts only grow, so this is where adding the next scene would first move the count further from
   the target.

**Result.** 24 whole scenes from 17 recordings. The far cars are the scored ground-truth cars (region and front-camera
view), counted by BEV range.

| Condition | fit scenes available | fit_sub frames | select frames | fit_sub scenes / recordings | select scenes / recordings | fit_sub cars 50–100 / 100–200 m | select cars 50–100 / 100–200 m |
|---|---|---|---|---|---|---|---|
| highway | 60 | 1,413 | 1,419 | 10 / 5 | 10 / 3 | 1,642 / 402 | 1,725 / 778 |
| urban | 44 | 1,133 | 1,136 | 8 / 7 | 8 / 4 | 3,553 / 779 | 3,454 / 1,271 |
| night | 21 | 566 | 569 | 4 / 4 | 4 / 3 | 570 / 259 | 1,289 / 488 |
| rain | 2 | 285 | 285 | 2 / 1 | 2 / 1 | 143 / 0 | 406 / 223 |
| **all** | 127 | **3,397** | **3,409** | 24 / 17 | 24 / 11 | **5,908 / 1,440** | **6,874 / 2,760** |

**What the subset does not equalize, known before training.**
- **Far cars.** `fit_sub` matches `select` in frames, but it has 86% of `select`'s cars at 50–100 m and 52% of its
  cars at 100–200 m. That is 7,348 against 9,634 cars at 50–200 m (76%).
  - `fit` has only 2 rain scenes. Both were drawn, and neither has a car beyond 100 m.
- **Diversity.** `fit_sub` spans more recordings than `select` (17 against 11).
- **Consequence.** A difference between CLOCs-R_sel and CLOCs-R_sub can come from the detectors having seen the scenes,
  or from `select` having about twice as many training cars at 100–200 m.
  - P1 (50–200 m) cannot separate those two by itself.
  - P2 is at 50–100 m, where the car counts are closest, so it is the cleaner test of the camera reliance.
- This caveat is fixed here and reported with the results, whichever way they come out.

## Order of work and val exposure

1. The subset is drawn (`subset.json`). This file and the code are committed.
2. All training. Each network is also scored on `select`, which no `fit_sub` scene overlaps. Those scores are
   recorded in `runs/fit_sub_control/` and not used for anything. No val scoring happens during training.
3. Val, **once per `_sub` model**, after every run has finished (`score_val.py`).
   - SECOND, the fit-trained models (L0, LRd, LC, CLOCs-R) and the select-trained primary models (L0_sel, LRd_sel,
     LC_sel, CLOCs-R_sel) are scored again from their saved networks in the same pass, so that every difference is
     seed-paired on the same resamples.
   - They must reproduce their stored val results: `results/lcr/data` and `results/select_trained/val.json`. That
     covers per-seed APCD, winning-cell counts, and the intervals of the comparisons those files hold.

## Evaluation (fixed; as results/select_trained)

- **Metric.** APCD:
  - centre-distance thresholds 0.5, 1, 2 and 4 m;
  - cars 0–204.8 m ahead, ±25.6 m to the side, in the front camera's view;
  - BEV NMS at 0.1.
- **Per model:** all, 0–30, 30–50, 50–100, 100–200 and 50–200 m, and 50–200 m by condition. Mean of seeds 1–6 and
  seed SD (ddof 1).
- **Bootstrap.** 1,000 resamples, `RandomState(0)` per level, seeds averaged inside each resample, percentile 95%
  intervals.
  - Two levels: scene (the verdicts use this level) and recording.
  - The point estimate is the difference of the 6-seed means, seed-paired.
- **Differences at 50–200, 50–100 and 100–200 m:**
  - CLOCs-R_sel − CLOCs-R_sub, LC_sel − LC_sub, L0_sel − L0_sub and LRd_sel − LRd_sub, plus L0_sub − L0_sel for P4;
  - CLOCs-R_sub − LC_sub and LRd_sub − L0_sub (the radar effects);
  - each `_sub` model minus SECOND;
  - each `_sub` model minus its full-`fit` counterpart.
- **Winning-cell shares** (`results/qualitative/`'s definition, as in results/select_trained):
  - CLOCs-R_sub next to CLOCs-R and CLOCs-R_sel, in 50–100 and 100–200 m;
  - 6-seed means, with seed 1 shown alongside;
  - scene and recording bootstraps of CLOCs-R_sub − CLOCs-R_sel and CLOCs-R_sub − CLOCs-R.

## Predictions and decision rules (val, 6 seeds, scene-level 95% interval)

**P1 (the camera reliance is about the scenes).** Δ = CLOCs-R_sel − CLOCs-R_sub at 50–200 m.
- Supported: Δ ≥ +0.005 and the interval's lower end > 0.
- Contradicted: the upper end < 0.
- Otherwise not supported, including an interval above 0 with Δ < +0.005.

**P2.** Δ = the camera cell's winning share in CLOCs-R_sub − its share in CLOCs-R_sel, at 50–100 m, using 6-seed
means.
- Supported: Δ > 0 and the lower end > 0.
- Contradicted: the upper end < 0.
- Otherwise not supported.

**P3.** Δ = CLOCs-R_sub − LC_sub at 50–200 m.
- Supported: Δ < +0.005, or the interval includes zero.
- Contradicted: Δ ≥ +0.005 and the interval excludes zero (its lower end > 0).
- No outcome falls between the two, so P3 has no "not supported".

**P4 (sensor-free control, reported in either direction).** Δ = L0_sub − L0_sel at 50–200 m.
- P4 predicts no direction, so it gets none of the three verdicts.
- It is reported as "L0_sub above L0_sel" (lower end > 0), "L0_sub below L0_sel" (upper end < 0), or "interval
  includes zero".

Numbers are taken from the result files and rounded to 3 decimals in the report. `make_report.py`, committed with
this file, applies these rules.

## Outputs

`results/fit_sub_control/`:
- `PREDICTIONS.md`, `subset.json`;
- `val.json` (every val number);
- `results.json`, `report_tables.md`, `REPORT.md`;
- the code: `draw_subset.py`, `run.sh`, `score_val.py`, `make_report.py`.

Training logs and networks go to `runs/fit_sub_control/`. Nothing else in `results/` is changed.
