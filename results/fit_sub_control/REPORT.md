# Size control for the select-trained result: report

The predictions, rules, subset and code were fixed in `PREDICTIONS.md` and committed in 3918d06, before any
network was trained on `fit_sub`. Training ran from 12:21 to 13:48 EST on 2026-11-27, and val from 13:48 to 16:48 EST.
None of the committed code or files changed afterwards.

Every number below is taken from `val.json`, `results.json`, `subset.json` or the training logs, rounded to 3
decimals. Intervals are 95% percentile intervals, at the scene level (the verdicts use this level) or the recording
level. Val was scored once, after all training had finished.

## Checks

- **Training frames.** Every `_sub` network trained on exactly `fit_sub`'s 3,397 frames, with 53,979 steps. The logs
  print `/53979` for every run.
- **Rescored models.** SECOND, the fit-trained models and the select-trained models were rescored from their saved
  networks in the same val pass:
  - **select-trained** (L0_sel, LRd_sel, LC_sel, CLOCs-R_sel): identical to `results/select_trained/val.json`. That
    covers 288 per-seed values per model, the winning-cell counts, and the scene and recording intervals of every
    comparison both files hold, including the share differences (largest difference 0.000).
  - **fit-trained** (L0, LRd, LC, CLOCs-R): their stored per-seed val results are reproduced to within 1.4e-6, the
    same as in results/select_trained.
  - **SECOND:** identical.
  - The stored scene intervals of CLOCs-R − LC, CLOCs-R − SECOND and LRd − L0 are identical.
  - Fit-trained CLOCs-R seed 1's winning-cell shares match `results/qualitative/` exactly.

## Verdicts

| | Test (val) | Δ | scene 95% | recording 95% | Verdict |
|---|---|---|---|---|---|
| **P1** | CLOCs-R_sel − CLOCs-R_sub, 50–200 m | +0.020 | [+0.011, +0.026] | [+0.000, +0.027] | **supported** |
| **P2** | camera share, CLOCs-R_sub − CLOCs-R_sel, 50–100 m | +0.288 | [+0.175, +0.410] | [+0.258, +0.485] | **supported** |
| **P3** | CLOCs-R_sub − LC_sub, 50–200 m | −0.006 | [−0.018, +0.004] | [−0.034, +0.003] | **supported** |
| **P4** | L0_sub − L0_sel, 50–200 m | +0.008 | [−0.000, +0.018] | [−0.001, +0.048] | interval includes zero (no direction predicted) |

- **P1.** Δ is above +0.005, and the scene interval is above zero.
  - The recording interval's lower end is +0.000 after rounding; its unrounded value is +0.00028.
  - At 50–100 m, Δ is +0.023 [+0.014, +0.030]. At 100–200 m it is +0.009 [−0.001, +0.018], an interval that includes
    zero.
- **P2.** CLOCs-R_sub's camera cell wins 83.2% of the matched candidates at 50–100 m, against 54.4% for CLOCs-R_sel.
  Both are 6-seed means.
  - That is close to fit-trained CLOCs-R's 88.3%: CLOCs-R_sub − CLOCs-R is −5.0 points [−8.1, −2.2].
  - At 100–200 m the shares are 58.6% (sub), 53.1% (sel) and 62.6% (fit). The difference sub − sel is +5.4 points
    [+3.2, +8.3].
- **P3.** On `fit_sub`, adding radar to the camera network does not help at 50–200 m (−0.006, interval includes zero).
  - At 100–200 m, CLOCs-R_sub − LC_sub is −0.016 [−0.033, −0.002].
  - On the select-trained side, the same comparison (H1) was +0.016 [+0.006, +0.027].
- **P4.** L0_sub − L0_sel is +0.008 at 50–200 m. Its scene interval just includes zero: the lower end is −0.0004.
  - At 100–200 m it is +0.043 [+0.024, +0.064].
  - L0_sub is close to the fit-trained L0: −0.001 [−0.003, +0.002] at 50–200 m.

## What this says, with the caveat fixed in advance

- **The result.** A `fit`-scene training set the size of `select` brings the camera reliance back:
  - the camera share is near the fit-trained network's;
  - radar adds nothing over the camera network;
  - CLOCs-R_sub − SECOND is −0.022 [−0.041, −0.004], close to the fit-trained −0.018.
- **On its face, the effect is about the scenes, not the size of the training set.** The networks trained on scenes
  the detectors saw lean on the camera; the networks trained on scenes no detector saw do not.
- **The caveat (PREDICTIONS.md).** `fit_sub` matches `select` in frames, but it has 52% of `select`'s cars at
  100–200 m (1,440 against 2,760) and 86% of its cars at 50–100 m.
  - P1 at 50–200 m therefore cannot, by itself, separate "the detectors saw the scenes" from "fewer far training cars".
  - P2 is at 50–100 m, where the car counts are closest, and its gap is the largest: 28.8 points. Car counts that
    differ by 14% are unlikely to explain that, but this control does not rule it out.
- **The other networks at 100–200 m (not pre-registered).** Trained on `select`, the networks without both sensors
  lose against their `fit_sub` counterparts at 100–200 m, with scene intervals below zero:
  - L0_sel − L0_sub −0.043;
  - LRd_sel − LRd_sub −0.025;
  - LC_sel − LC_sub −0.029.

  Meanwhile, every `_sub` network stays within ±0.008 of its full-`fit` counterpart at 100–200 m. So the 100–200 m
  losses of the select-trained models (results/select_trained) are also not about the size of the training set; they
  come with `select`'s scenes. Note that this includes L0, which has no camera or radar cell, so it is not only about
  the camera.

## Tables

### The subset: frames, recordings and far cars (scored cars, BEV range), against select and fit

| Condition | fit_sub frames | select frames | fit_sub scenes / recordings | select scenes / recordings | fit_sub cars 50–100 / 100–200 m | select cars 50–100 / 100–200 m |
|---|---|---|---|---|---|---|
| highway | 1,413 | 1,419 | 10 / 5 | 10 / 3 | 1,642 / 402 | 1,725 / 778 |
| urban | 1,133 | 1,136 | 8 / 7 | 8 / 4 | 3,553 / 779 | 3,454 / 1,271 |
| night | 566 | 569 | 4 / 4 | 4 / 3 | 570 / 259 | 1,289 / 488 |
| rain | 285 | 285 | 2 / 1 | 2 / 1 | 143 / 0 | 406 / 223 |
| all | 3,397 | 3,409 | 24 / 17 | 24 / 11 | 5,908 / 1,440 | 6,874 / 2,760 |

### Per model, val APCD (mean ± seed SD, seeds 1–6)

| Model | all | 0–30 | 30–50 | 50–100 | 100–200 | 50–200 |
|---|---|---|---|---|---|---|
| SECOND | 0.540 | 0.897 | 0.621 | 0.599 | 0.284 | 0.446 |
| L0_sub | 0.501 ± 0.001 | 0.895 ± 0.001 | 0.620 ± 0.000 | 0.595 ± 0.000 | 0.281 ± 0.001 | 0.427 ± 0.001 |
| L0_sel | 0.502 ± 0.006 | 0.898 ± 0.001 | 0.620 ± 0.000 | 0.597 ± 0.001 | 0.238 ± 0.019 | 0.419 ± 0.009 |
| L0 | 0.502 ± 0.001 | 0.896 ± 0.000 | 0.620 ± 0.000 | 0.598 ± 0.000 | 0.273 ± 0.002 | 0.428 ± 0.001 |
| LRd_sub | 0.514 ± 0.002 | 0.891 ± 0.001 | 0.621 ± 0.001 | 0.602 ± 0.001 | 0.290 ± 0.002 | 0.447 ± 0.002 |
| LRd_sel | 0.513 ± 0.002 | 0.890 ± 0.001 | 0.625 ± 0.001 | 0.605 ± 0.001 | 0.265 ± 0.005 | 0.435 ± 0.002 |
| LRd | 0.513 ± 0.002 | 0.891 ± 0.001 | 0.621 ± 0.001 | 0.603 ± 0.000 | 0.286 ± 0.001 | 0.442 ± 0.002 |
| LC_sub | 0.529 ± 0.001 | 0.916 ± 0.001 | 0.626 ± 0.001 | 0.572 ± 0.002 | 0.285 ± 0.002 | 0.430 ± 0.001 |
| LC_sel | 0.531 ± 0.005 | 0.916 ± 0.001 | 0.642 ± 0.001 | 0.592 ± 0.003 | 0.256 ± 0.011 | 0.427 ± 0.006 |
| LC | 0.525 ± 0.002 | 0.916 ± 0.001 | 0.630 ± 0.001 | 0.577 ± 0.002 | 0.279 ± 0.004 | 0.426 ± 0.002 |
| CLOCs-R_sub | 0.524 ± 0.005 | 0.912 ± 0.001 | 0.633 ± 0.002 | 0.579 ± 0.004 | 0.270 ± 0.010 | 0.423 ± 0.006 |
| CLOCs-R_sel | 0.537 ± 0.006 | 0.907 ± 0.001 | 0.639 ± 0.001 | 0.602 ± 0.004 | 0.279 ± 0.010 | 0.443 ± 0.008 |
| CLOCs-R | 0.525 ± 0.001 | 0.913 ± 0.004 | 0.629 ± 0.003 | 0.580 ± 0.004 | 0.278 ± 0.007 | 0.427 ± 0.002 |

### Per model, val APCD at 50–200 m by condition (mean ± seed SD)

| Model | highway | urban | night | rain |
|---|---|---|---|---|
| SECOND | 0.434 | 0.448 | 0.428 | 0.549 |
| L0_sub | 0.414 ± 0.001 | 0.428 ± 0.000 | 0.417 ± 0.000 | 0.528 ± 0.001 |
| L0_sel | 0.364 ± 0.025 | 0.430 ± 0.006 | 0.408 ± 0.009 | 0.530 ± 0.008 |
| L0 | 0.403 ± 0.003 | 0.432 ± 0.001 | 0.416 ± 0.001 | 0.527 ± 0.001 |
| LRd_sub | 0.445 ± 0.003 | 0.442 ± 0.002 | 0.445 ± 0.001 | 0.561 ± 0.003 |
| LRd_sel | 0.388 ± 0.008 | 0.436 ± 0.002 | 0.442 ± 0.002 | 0.557 ± 0.002 |
| LRd | 0.425 ± 0.004 | 0.440 ± 0.002 | 0.436 ± 0.001 | 0.549 ± 0.002 |
| LC_sub | 0.448 ± 0.003 | 0.418 ± 0.001 | 0.443 ± 0.002 | 0.568 ± 0.005 |
| LC_sel | 0.392 ± 0.016 | 0.435 ± 0.004 | 0.423 ± 0.008 | 0.555 ± 0.006 |
| LC | 0.429 ± 0.005 | 0.421 ± 0.001 | 0.437 ± 0.004 | 0.547 ± 0.007 |
| CLOCs-R_sub | 0.420 ± 0.012 | 0.416 ± 0.005 | 0.440 ± 0.005 | 0.559 ± 0.015 |
| CLOCs-R_sel | 0.418 ± 0.022 | 0.438 ± 0.003 | 0.462 ± 0.007 | 0.578 ± 0.007 |
| CLOCs-R | 0.427 ± 0.006 | 0.421 ± 0.002 | 0.436 ± 0.007 | 0.553 ± 0.011 |

### Seed-paired differences on val at 50–200 m (95% intervals)

| Difference | Δ | scene | recording |
|---|---|---|---|
| CLOCs-R_sel − CLOCs-R_sub | +0.020 | [+0.011, +0.026] | [+0.000, +0.027] |
| LC_sel − LC_sub | -0.002 | [-0.020, +0.009] | [-0.054, +0.013] |
| L0_sel − L0_sub | -0.008 | [-0.018, +0.000] | [-0.048, +0.001] |
| LRd_sel − LRd_sub | -0.012 | [-0.024, -0.002] | [-0.057, +0.001] |
| L0_sub − L0_sel | +0.008 | [-0.000, +0.018] | [-0.001, +0.048] |
| CLOCs-R_sub − LC_sub | -0.006 | [-0.018, +0.004] | [-0.034, +0.003] |
| LRd_sub − L0_sub | +0.020 | [+0.010, +0.030] | [+0.015, +0.037] |
| L0_sub − SECOND | -0.018 | [-0.024, -0.013] | [-0.026, -0.010] |
| LRd_sub − SECOND | +0.001 | [-0.009, +0.011] | [-0.004, +0.014] |
| LC_sub − SECOND | -0.016 | [-0.028, -0.001] | [-0.027, +0.020] |
| CLOCs-R_sub − SECOND | -0.022 | [-0.041, -0.004] | [-0.043, +0.020] |
| L0_sub − L0 | -0.001 | [-0.003, +0.002] | [-0.004, +0.010] |
| LRd_sub − LRd | +0.005 | [+0.002, +0.010] | [+0.002, +0.022] |
| LC_sub − LC | +0.003 | [-0.002, +0.011] | [-0.002, +0.023] |
| CLOCs-R_sub − CLOCs-R | -0.004 | [-0.009, +0.002] | [-0.011, +0.008] |

### Seed-paired differences on val at 50–100 m (95% intervals)

| Difference | Δ | scene | recording |
|---|---|---|---|
| CLOCs-R_sel − CLOCs-R_sub | +0.023 | [+0.014, +0.030] | [+0.003, +0.030] |
| LC_sel − LC_sub | +0.021 | [+0.008, +0.030] | [-0.008, +0.031] |
| L0_sel − L0_sub | +0.003 | [-0.001, +0.008] | [-0.001, +0.006] |
| LRd_sel − LRd_sub | +0.003 | [-0.004, +0.009] | [-0.002, +0.008] |
| L0_sub − L0_sel | -0.003 | [-0.008, +0.001] | [-0.006, +0.001] |
| CLOCs-R_sub − LC_sub | +0.007 | [-0.004, +0.022] | [-0.007, +0.010] |
| LRd_sub − L0_sub | +0.007 | [-0.007, +0.025] | [+0.002, +0.021] |
| L0_sub − SECOND | -0.005 | [-0.008, -0.003] | [-0.006, -0.000] |
| LRd_sub − SECOND | +0.003 | [-0.013, +0.019] | [-0.003, +0.017] |
| LC_sub − SECOND | -0.028 | [-0.043, -0.011] | [-0.040, +0.006] |
| CLOCs-R_sub − SECOND | -0.020 | [-0.040, +0.003] | [-0.032, +0.014] |
| L0_sub − L0 | -0.003 | [-0.005, -0.002] | [-0.004, -0.000] |
| LRd_sub − LRd | -0.001 | [-0.002, +0.001] | [-0.002, +0.004] |
| LC_sub − LC | -0.005 | [-0.009, +0.002] | [-0.009, +0.010] |
| CLOCs-R_sub − CLOCs-R | -0.001 | [-0.008, +0.008] | [-0.004, +0.009] |

### Seed-paired differences on val at 100–200 m (95% intervals)

| Difference | Δ | scene | recording |
|---|---|---|---|
| CLOCs-R_sel − CLOCs-R_sub | +0.009 | [-0.001, +0.018] | [-0.008, +0.022] |
| LC_sel − LC_sub | -0.029 | [-0.044, -0.016] | [-0.073, -0.014] |
| L0_sel − L0_sub | -0.043 | [-0.064, -0.024] | [-0.104, -0.022] |
| LRd_sel − LRd_sub | -0.025 | [-0.046, -0.009] | [-0.087, -0.002] |
| L0_sub − L0_sel | +0.043 | [+0.024, +0.064] | [+0.022, +0.104] |
| CLOCs-R_sub − LC_sub | -0.016 | [-0.033, -0.002] | [-0.052, +0.005] |
| LRd_sub − L0_sub | +0.009 | [+0.003, +0.016] | [+0.007, +0.018] |
| L0_sub − SECOND | -0.003 | [-0.005, -0.001] | [-0.006, +0.001] |
| LRd_sub − SECOND | +0.006 | [-0.001, +0.014] | [+0.002, +0.018] |
| LC_sub − SECOND | +0.001 | [-0.013, +0.018] | [-0.015, +0.035] |
| CLOCs-R_sub − SECOND | -0.015 | [-0.042, +0.013] | [-0.063, +0.036] |
| L0_sub − L0 | +0.008 | [+0.002, +0.014] | [+0.001, +0.027] |
| LRd_sub − LRd | +0.004 | [-0.002, +0.011] | [-0.002, +0.025] |
| LC_sub − LC | +0.006 | [+0.001, +0.013] | [+0.002, +0.021] |
| CLOCs-R_sub − CLOCs-R | -0.008 | [-0.017, -0.000] | [-0.026, +0.004] |

### Winning cell at SECOND candidates matched to val cars (share, %)

| Model | bin | candidates | camera | radar | reserved | camera, seed 1 |
|---|---|---|---|---|---|---|
| CLOCs-R | 50-100 m | 7,841 | 88.3 | 9.2 | 2.5 | 82.9 |
| CLOCs-R | 100-200 m | 7,296 | 62.6 | 7.5 | 29.9 | 58.8 |
| CLOCs-R_sel | 50-100 m | 7,841 | 54.4 | 43.1 | 2.5 | 44.3 |
| CLOCs-R_sel | 100-200 m | 7,296 | 53.1 | 16.9 | 29.9 | 49.2 |
| CLOCs-R_sub | 50-100 m | 7,841 | 83.2 | 14.2 | 2.5 | 85.3 |
| CLOCs-R_sub | 100-200 m | 7,296 | 58.6 | 11.5 | 29.9 | 60.3 |

| Share difference (points) | bin | Δ | scene | recording |
|---|---|---|---|---|
| CLOCs-R_sub − CLOCs-R_sel, camera | 50-100 m | +28.8 | [+17.5, +41.0] | [+25.8, +48.5] |
| CLOCs-R_sub − CLOCs-R_sel, radar | 50-100 m | -28.8 | [-41.0, -17.5] | [-48.5, -25.8] |
| CLOCs-R_sub − CLOCs-R_sel, reserved | 50-100 m | +0.0 | [+0.0, +0.0] | [+0.0, +0.0] |
| CLOCs-R_sub − CLOCs-R_sel, camera | 100-200 m | +5.4 | [+3.2, +8.3] | [+3.3, +10.6] |
| CLOCs-R_sub − CLOCs-R_sel, radar | 100-200 m | -5.4 | [-8.3, -3.2] | [-10.6, -3.3] |
| CLOCs-R_sub − CLOCs-R_sel, reserved | 100-200 m | +0.0 | [+0.0, +0.0] | [+0.0, +0.0] |
| CLOCs-R_sub − CLOCs-R, camera | 50-100 m | -5.0 | [-8.1, -2.2] | [-8.7, -2.2] |
| CLOCs-R_sub − CLOCs-R, radar | 50-100 m | +5.0 | [+2.2, +8.1] | [+2.2, +8.7] |
| CLOCs-R_sub − CLOCs-R, reserved | 50-100 m | +0.0 | [+0.0, +0.0] | [+0.0, +0.0] |
| CLOCs-R_sub − CLOCs-R, camera | 100-200 m | -4.0 | [-7.1, -1.4] | [-8.5, -1.4] |
| CLOCs-R_sub − CLOCs-R, radar | 100-200 m | +4.0 | [+1.4, +7.1] | [+1.4, +8.5] |
| CLOCs-R_sub − CLOCs-R, reserved | 100-200 m | +0.0 | [+0.0, +0.0] | [+0.0, +0.0] |
| CLOCs-R_sel − CLOCs-R, camera | 50-100 m | -33.8 | [-48.2, -19.8] | [-52.2, -31.1] |
| CLOCs-R_sel − CLOCs-R, radar | 50-100 m | +33.8 | [+19.8, +48.2] | [+31.1, +52.2] |
| CLOCs-R_sel − CLOCs-R, reserved | 50-100 m | +0.0 | [+0.0, +0.0] | [+0.0, +0.0] |
| CLOCs-R_sel − CLOCs-R, camera | 100-200 m | -9.4 | [-14.6, -5.1] | [-18.8, -5.5] |
| CLOCs-R_sel − CLOCs-R, radar | 100-200 m | +9.4 | [+5.1, +14.6] | [+5.5, +18.8] |
| CLOCs-R_sel − CLOCs-R, reserved | 100-200 m | +0.0 | [+0.0, +0.0] | [+0.0, +0.0] |

## Training time

Times are wall-clock on one shared GPU, with three seeds training at a time. They come from
`runs/fit_sub_control/times.tsv`, in EST, on 2026-11-27.

| Step | Wall time |
|---|---|
| L0_sub, 6 seeds × 53,979 steps | 20.1 min |
| LRd_sub | 19.6 min |
| LC_sub | 23.0 min |
| CLOCs-R_sub | 24.1 min |
| **All training (12:21–13:48)** | **86.8 min** |
| Val scoring: 73 model-seeds, with both bootstrap levels (13:48–16:48) | 179.4 min |

Each run loads all of `fit`'s cached cells and then keeps the 3,397 `fit_sub` frames. That is why a batch of three
seeds takes 10–12 min here, against 10–11 min on `select`.

Disk: 4.6 GB free at the start and at the end.

## Deviations and choices the prompt did not fix

1. **The commit.** It was made by hand (3918d06), outside the automated session, because the session's permission
   settings refused the commit command. `run.sh` checks that `PREDICTIONS.md` and `subset.json` are committed and
   unchanged before it trains.
2. **Code.** `--train_scenes` was added to `clocs_r/tune_fusion.py`. Its default leaves every existing run unchanged.
   - It filters `fit`'s cached frames after loading, rather than building a `fit_sub` cache. `fit_base.pkl` is 4.9 GB,
     and the disk has 4.6 GB free.
   - The filter keeps the cached order, so the training frames are the same list a separate cache would hold.
3. **Subset draw.**
   - A fresh `RandomState(0)` per condition.
   - The stop is at the first prefix of the shuffle whose kept-frame count is closest to `select`'s.
   - Far cars are the scored ground-truth cars (region and front-camera view), counted by BEV range in the metric's
     bins.
4. **P3 and P4 outcomes.** P3's rule ("below +0.005, or the interval includes zero") allows only supported or
   contradicted. P4 has no predicted direction, so it is reported as above, below, or "interval includes zero".
5. **No B_sub.** The prompt does not ask for one.
6. **Rescoring.** The fit-trained and select-trained networks were scored again in the same val pass, for seed pairing.
   They reproduce their stored results (see the checks above). The `_sub` networks were scored on val once.
7. **Scoring on `select` during training.** Each `_sub` network was also scored on `select` during training, because
   `tune_fusion` evaluates some split. These scores are recorded in `runs/fit_sub_control/train_primary.json` and not
   used.
8. **The recorded step count.** The per-seed `steps` field in the training JSON reads 53,800, the last logged step
   (the loss is logged every 200 steps). The schedule and the updates are 53,979, as in results/select_trained.
9. **Winning-cell shares.** They are the mean of the six per-seed shares, with seed 1 alongside. Shares are given in
   percent with one decimal, which is 3 decimals as a fraction.

Nothing else went differently from `PREDICTIONS.md`. `make_report.py` ran as committed. No setting was tuned, and
nothing outside `results/fit_sub_control/` was written in `results/`.

## Files

- `PREDICTIONS.md` and `subset.json`: fixed before training.
- `val.json`: every val number, from `score_val.py`.
- `results.json`: the subset, verdicts, tables, training and reproduction, from `make_report.py`.
- `report_tables.md`: the tables above, as generated.
- The code: `draw_subset.py`, `run.sh`, `score_val.py` and `make_report.py`.
- The networks and logs are in `runs/fit_sub_control/`. They are not tracked.
