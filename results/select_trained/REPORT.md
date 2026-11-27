# Fusion networks trained on `select`: report

The predictions, rules and settings were fixed in `PREDICTIONS.md`. That file was committed in f4b3232 at 13:20:06 EST
on 2026-11-26, before any network was trained on `select`; the first one started at 13:20:26 EST. Every number below
is taken from `val.json`, `results.json`, `step0.json`, `equivalence.json` or the training logs, rounded to 3 decimals.
Intervals are 95% percentile intervals at the scene level (the level the verdicts use) or the recording level.
Val was scored once, after all training had finished.

## Checks before the results

- **Step 0.** Fit-trained CLOCs-R (`R0+T2+K40+RC`) seed 1 was retrained with the current code.
  - Val APCD at 50–200 m: 0.426, against the stored 0.426 (`results/lcr/data/lcr_val.json`).
  - The difference is 0.000 in every distance bin and condition. The two values are bit-identical.
- **Equivalence.** The thesis code and master's code (NC token, `--steps`, `blend.py`) each trained the same networks:
  LC, L0, LRd, CLOCs-R, each with and without `--steps`, and the blend, on the same 200 `select` frames.
  - All weights were identical (largest difference 0.000).
  - A run asked for 300 updates made exactly 300.
- **Fit-trained models, rescored from their saved networks in the same val pass.** SECOND, L0, LRd, LC, CLOCs-R and B,
  seeds 1–6, against their stored per-seed val results (288 values per model, 48 for SECOND).
  - SECOND and B: identical.
  - The others: largest difference 1.4e-6, from computing the sigmoid in torch rather than numpy.
  - The stored scene-level intervals of CLOCs-R − LC, CLOCs-R − SECOND, LRd − L0 and B − LC: identical.
  - Fit-trained CLOCs-R seed 1's winning-cell shares match `results/qualitative/` exactly: 7,841 candidates with
    82.9% camera, and 7,296 with 58.8% camera.

## Verdicts (primary length: 53,979 steps)

| | Test (val, 50–200 m unless stated) | Δ | scene 95% | recording 95% | Verdict |
|---|---|---|---|---|---|
| **H1** | CLOCs-R_sel − LC_sel | +0.016 | [+0.006, +0.027] | [+0.007, +0.039] | **supported** |
| **H2** | camera share, CLOCs-R_sel − CLOCs-R, 50–100 m | −0.338 | [−0.482, −0.198] | [−0.522, −0.311] | **supported** |
| | camera share, CLOCs-R_sel − CLOCs-R, 100–200 m | −0.094 | [−0.146, −0.051] | [−0.188, −0.055] | |
| **H3** | CLOCs-R_sel − CLOCs-R | +0.016 | [+0.007, +0.024] | [−0.002, +0.026] | **supported** |
| **H4** | L0_sel − L0 | −0.009 | [−0.017, −0.003] | [−0.038, −0.002] | **contradicted** |

- **H1.** The fit-trained value was +0.001 [−0.004, +0.005].
  - CLOCs-R_sel − LC_sel is +0.016. That is above +0.005, and the scene interval is above zero.
  - At 100–200 m it is +0.022 [+0.014, +0.034].
- **H2.** CLOCs-R_sel's camera share, as the mean of the six per-seed shares:
  - 50–100 m: 54.4%, against 88.3% for fit-trained CLOCs-R (82.9% for seed 1).
  - 100–200 m: 53.1%, against 62.6% (58.8% for seed 1).
  - Both differences and their scene intervals are below zero. The radar cell took over exactly the share the camera
    lost: 9.2% → 43.1% and 7.5% → 16.9%.
- **H3.** Fit-trained CLOCs-R − SECOND was −0.018 [−0.033, −0.005]. CLOCs-R_sel − SECOND is −0.003
  [−0.019, +0.012].
  - The seed-paired change is +0.016 [+0.007, +0.024].
  - The interval of CLOCs-R_sel − SECOND includes zero, so it **does not exclude zero in either direction**.
  - At the recording level, the H3 interval [−0.002, +0.026] includes zero.
- **H4.** Fit-trained L0 − SECOND was −0.017 [−0.025, −0.011]. L0_sel − SECOND is −0.026 [−0.041, −0.016].
  - That is further from zero, not closer.
  - The seed-paired L0_sel − L0 is −0.009, and its scene interval is entirely below zero, so H4 is contradicted.

**Secondary (3 epochs over `select`, 10,227 steps).** These are labeled secondary and are not the verdicts that count.

| | Test (val, 50–200 m) | Δ | scene 95% | recording 95% | Verdict |
|---|---|---|---|---|---|
| H1 | CLOCs-R_sel3 − LC_sel3 | +0.007 | [−0.005, +0.019] | [−0.007, +0.030] | not supported |
| H3 | CLOCs-R_sel3 − CLOCs-R | +0.021 | [+0.012, +0.029] | [+0.010, +0.032] | supported |
| | CLOCs-R_sel3 − SECOND (reported) | +0.003 | [−0.015, +0.019] | [−0.013, +0.036] | interval includes zero |

## Reading the results (not part of the pre-registered tests)

- **CLOCs-R_sel's gain sits at 50–100 m.** Against fit-trained CLOCs-R:
  - 50–100 m: +0.022 [+0.010, +0.033] (`results.json`, `differences`).
  - 100–200 m: +0.001 [−0.007, +0.009].
- **At 100–200 m, H1 is large partly because the networks without radar got worse.** Training on `select` cost every
  select-trained model except CLOCs-R_sel at 100–200 m, with scene intervals entirely below zero:
  - L0_sel −0.035;
  - LRd_sel −0.021;
  - LC_sel −0.022;
  - B_sel −0.022.
  At 50–200 m the picture is different: LC_sel − LC is +0.001 [−0.011, +0.009], so H1's +0.016 at 50–200 m comes from
  CLOCs-R_sel's own gain.
- **H4.** The loss of the sensor-free control is concentrated at 100–200 m and on highway:
  - 100–200 m: 0.238 ± 0.019 against 0.273 ± 0.002 for L0.
  - Highway: 0.364 ± 0.025 against 0.403 ± 0.003.
  - The spread across seeds is also much larger than for the fit-trained L0.
- **The primary length may be too long for 3,409 frames.** It amounts to 15.83 passes over `select`. All five models
  score higher at 50–200 m with the secondary 3-epoch length than with the primary length (secondary table below).
  - These are point means only; no interval was computed for this comparison, and it was not pre-registered.
- **B.**
  - B_sel − B is −0.002 [−0.013, +0.006] at 50–200 m.
  - B_sel − CLOCs-R_sel is −0.002 [−0.013, +0.006] at 50–200 m and −0.010 [−0.020, −0.002] at 100–200 m.
  - B_sel moves weight from LC to LRd (LC / LRd 0.561 / 0.439, against B's 0.723 / 0.277). SECOND's weight stays 0.000
    in the primary runs and 0.007 in the secondary runs.

## Tables

### Per model, val APCD (mean ± seed SD, seeds 1–6)

| Model | all | 0–30 | 30–50 | 50–100 | 100–200 | 50–200 |
|---|---|---|---|---|---|---|
| SECOND |0.540 | 0.897 | 0.621 | 0.599 | 0.284 | 0.446 |
| L0_sel |0.502 ± 0.006 | 0.898 ± 0.001 | 0.620 ± 0.000 | 0.597 ± 0.001 | 0.238 ± 0.019 | 0.419 ± 0.009 |
| L0 |0.502 ± 0.001 | 0.896 ± 0.000 | 0.620 ± 0.000 | 0.598 ± 0.000 | 0.273 ± 0.002 | 0.428 ± 0.001 |
| LRd_sel |0.513 ± 0.002 | 0.890 ± 0.001 | 0.625 ± 0.001 | 0.605 ± 0.001 | 0.265 ± 0.005 | 0.435 ± 0.002 |
| LRd |0.513 ± 0.002 | 0.891 ± 0.001 | 0.621 ± 0.001 | 0.603 ± 0.000 | 0.286 ± 0.001 | 0.442 ± 0.002 |
| LC_sel |0.531 ± 0.005 | 0.916 ± 0.001 | 0.642 ± 0.001 | 0.592 ± 0.003 | 0.256 ± 0.011 | 0.427 ± 0.006 |
| LC |0.525 ± 0.002 | 0.916 ± 0.001 | 0.630 ± 0.001 | 0.577 ± 0.002 | 0.279 ± 0.004 | 0.426 ± 0.002 |
| CLOCs-R_sel |0.537 ± 0.006 | 0.907 ± 0.001 | 0.639 ± 0.001 | 0.602 ± 0.004 | 0.279 ± 0.010 | 0.443 ± 0.008 |
| CLOCs-R |0.525 ± 0.001 | 0.913 ± 0.004 | 0.629 ± 0.003 | 0.580 ± 0.004 | 0.278 ± 0.007 | 0.427 ± 0.002 |
| B_sel |0.538 ± 0.003 | 0.914 ± 0.001 | 0.644 ± 0.001 | 0.610 ± 0.001 | 0.269 ± 0.006 | 0.440 ± 0.003 |
| B |0.538 ± 0.002 | 0.918 ± 0.001 | 0.640 ± 0.002 | 0.599 ± 0.002 | 0.291 ± 0.002 | 0.443 ± 0.002 |

### Per model, val APCD at 50–200 m by condition (mean ± seed SD)

| Model | highway | urban | night | rain |
|---|---|---|---|---|
| SECOND |0.434 | 0.448 | 0.428 | 0.549 |
| L0_sel |0.364 ± 0.025 | 0.430 ± 0.006 | 0.408 ± 0.009 | 0.530 ± 0.008 |
| L0 |0.403 ± 0.003 | 0.432 ± 0.001 | 0.416 ± 0.001 | 0.527 ± 0.001 |
| LRd_sel |0.388 ± 0.008 | 0.436 ± 0.002 | 0.442 ± 0.002 | 0.557 ± 0.002 |
| LRd |0.425 ± 0.004 | 0.440 ± 0.002 | 0.436 ± 0.001 | 0.549 ± 0.002 |
| LC_sel |0.392 ± 0.016 | 0.435 ± 0.004 | 0.423 ± 0.008 | 0.555 ± 0.006 |
| LC |0.429 ± 0.005 | 0.421 ± 0.001 | 0.437 ± 0.004 | 0.547 ± 0.007 |
| CLOCs-R_sel |0.418 ± 0.022 | 0.438 ± 0.003 | 0.462 ± 0.007 | 0.578 ± 0.007 |
| CLOCs-R |0.427 ± 0.006 | 0.421 ± 0.002 | 0.436 ± 0.007 | 0.553 ± 0.011 |
| B_sel |0.394 ± 0.009 | 0.447 ± 0.002 | 0.441 ± 0.003 | 0.564 ± 0.004 |
| B |0.436 ± 0.003 | 0.440 ± 0.001 | 0.444 ± 0.003 | 0.554 ± 0.006 |

### Seed-paired differences on val (scene and recording 95% intervals)

| Difference | 50–200 m | scene | recording | 100–200 m | scene | recording |
|---|---|---|---|---|---|---|
| L0_sel − L0 | -0.009 | [-0.017, -0.003] | [-0.038, -0.002] | -0.035 | [-0.050, -0.021] | [-0.077, -0.021] |
| LRd_sel − LRd | -0.007 | [-0.016, +0.000] | [-0.036, +0.009] | -0.021 | [-0.036, -0.009] | [-0.062, -0.002] |
| LC_sel − LC | +0.001 | [-0.011, +0.009] | [-0.033, +0.012] | -0.022 | [-0.036, -0.011] | [-0.055, -0.011] |
| CLOCs-R_sel − CLOCs-R | +0.016 | [+0.007, +0.024] | [-0.002, +0.026] | +0.001 | [-0.007, +0.009] | [-0.017, +0.010] |
| B_sel − B | -0.002 | [-0.013, +0.006] | [-0.038, +0.007] | -0.022 | [-0.035, -0.012] | [-0.063, -0.009] |
| CLOCs-R_sel − LC_sel | +0.016 | [+0.006, +0.027] | [+0.007, +0.039] | +0.022 | [+0.014, +0.034] | [+0.016, +0.046] |
| LRd_sel − L0_sel | +0.015 | [+0.004, +0.027] | [+0.009, +0.032] | +0.027 | [+0.016, +0.038] | [+0.022, +0.044] |
| LC_sel − L0_sel | +0.008 | [-0.002, +0.018] | [+0.004, +0.021] | +0.018 | [+0.007, +0.031] | [+0.013, +0.034] |
| CLOCs-R_sel − LRd_sel | +0.008 | [+0.001, +0.017] | [+0.004, +0.025] | +0.014 | [+0.006, +0.025] | [+0.008, +0.032] |
| L0_sel − SECOND | -0.026 | [-0.041, -0.016] | [-0.073, -0.015] | -0.046 | [-0.066, -0.029] | [-0.103, -0.026] |
| LRd_sel − SECOND | -0.011 | [-0.027, +0.003] | [-0.046, +0.013] | -0.019 | [-0.040, -0.002] | [-0.070, +0.010] |
| LC_sel − SECOND | -0.018 | [-0.033, -0.007] | [-0.057, -0.005] | -0.028 | [-0.050, -0.009] | [-0.080, -0.002] |
| CLOCs-R_sel − SECOND | -0.003 | [-0.019, +0.012] | [-0.026, +0.030] | -0.005 | [-0.029, +0.015] | [-0.048, +0.033] |
| B_sel − SECOND | -0.005 | [-0.021, +0.006] | [-0.046, +0.010] | -0.015 | [-0.036, +0.002] | [-0.068, +0.010] |
| B_sel − CLOCs-R_sel | -0.002 | [-0.013, +0.006] | [-0.023, +0.006] | -0.010 | [-0.020, -0.002] | [-0.028, -0.004] |
| CLOCs-R_sel (3 ep.) − LC_sel (3 ep.) | +0.007 | [-0.005, +0.019] | [-0.007, +0.030] | +0.002 | [-0.010, +0.015] | [-0.020, +0.024] |
| CLOCs-R_sel (3 ep.) − CLOCs-R | +0.021 | [+0.012, +0.029] | [+0.010, +0.032] | +0.009 | [+0.002, +0.017] | [+0.001, +0.020] |
| CLOCs-R_sel (3 ep.) − SECOND | +0.003 | [-0.015, +0.019] | [-0.013, +0.036] | +0.003 | [-0.020, +0.023] | [-0.029, +0.045] |

### Winning cell at SECOND candidates matched to val cars (share, %)

| Model | bin | candidates | camera | radar | reserved | camera, seed 1 |
|---|---|---|---|---|---|---|
| CLOCs-R | 50-100 m | 7,841 | 88.3 | 9.2 | 2.5 | 82.9 |
| CLOCs-R | 100-200 m | 7,296 | 62.6 | 7.5 | 29.9 | 58.8 |
| CLOCs-R_sel | 50-100 m | 7,841 | 54.4 | 43.1 | 2.5 | 44.3 |
| CLOCs-R_sel | 100-200 m | 7,296 | 53.1 | 16.9 | 29.9 | 49.2 |
| CLOCs-R_sel (3 ep.) | 50-100 m | 7,841 | 44.7 | 52.7 | 2.5 | 44.3 |
| CLOCs-R_sel (3 ep.) | 100-200 m | 7,296 | 48.8 | 21.3 | 29.9 | 48.6 |

| Share difference (camera / radar / reserved, points) | bin | difference | scene | recording |
|---|---|---|---|---|
| CLOCs-R_sel − CLOCs-R, camera | 50-100 m | -33.8 | [-48.2, -19.8] | [-52.2, -31.1] |
| CLOCs-R_sel − CLOCs-R, radar | 50-100 m | +33.8 | [+19.8, +48.2] | [+31.1, +52.2] |
| CLOCs-R_sel − CLOCs-R, reserved | 50-100 m | +0.0 | [+0.0, +0.0] | [+0.0, +0.0] |
| CLOCs-R_sel − CLOCs-R, camera | 100-200 m | -9.4 | [-14.6, -5.1] | [-18.8, -5.5] |
| CLOCs-R_sel − CLOCs-R, radar | 100-200 m | +9.4 | [+5.1, +14.6] | [+5.5, +18.8] |
| CLOCs-R_sel − CLOCs-R, reserved | 100-200 m | +0.0 | [+0.0, +0.0] | [+0.0, +0.0] |
| CLOCs-R_sel (3 ep.) − CLOCs-R, camera | 50-100 m | -43.5 | [-60.8, -27.5] | [-72.8, -39.5] |
| CLOCs-R_sel (3 ep.) − CLOCs-R, radar | 50-100 m | +43.5 | [+27.5, +60.8] | [+39.5, +72.8] |
| CLOCs-R_sel (3 ep.) − CLOCs-R, reserved | 50-100 m | +0.0 | [+0.0, +0.0] | [+0.0, +0.0] |
| CLOCs-R_sel (3 ep.) − CLOCs-R, camera | 100-200 m | -13.8 | [-20.8, -8.0] | [-28.1, -8.2] |
| CLOCs-R_sel (3 ep.) − CLOCs-R, radar | 100-200 m | +13.8 | [+8.0, +20.8] | [+8.2, +28.1] |
| CLOCs-R_sel (3 ep.) − CLOCs-R, reserved | 100-200 m | +0.0 | [+0.0, +0.0] | [+0.0, +0.0] |

### Blend weights (softmax, L / LC / LRd)

| Blend | seed 1 | 2 | 3 | 4 | 5 | 6 | mean |
|---|---|---|---|---|---|---|---|
| B_sel primary | 0.000 / 0.558 / 0.442 | 0.000 / 0.556 / 0.444 | 0.000 / 0.553 / 0.447 | 0.000 / 0.566 / 0.434 | 0.000 / 0.557 / 0.443 | 0.000 / 0.576 / 0.424 | 0.000 / 0.561 / 0.439 |
| B_sel secondary | 0.006 / 0.574 / 0.420 | 0.006 / 0.584 / 0.410 | 0.007 / 0.548 / 0.445 | 0.007 / 0.571 / 0.422 | 0.007 / 0.550 / 0.443 | 0.007 / 0.496 / 0.497 | 0.007 / 0.554 / 0.439 |
| B (fit-trained) | 0.000 / 0.717 / 0.283 | 0.000 / 0.722 / 0.278 | 0.000 / 0.723 / 0.277 | 0.000 / 0.718 / 0.282 | 0.000 / 0.740 / 0.260 | 0.000 / 0.718 / 0.282 | 0.000 / 0.723 / 0.277 |

### Secondary (3 epochs over select): val APCD at 50–200 m

| Model | mean ± SD | primary length | fit-trained |
|---|---|---|---|
| L0_sel | 0.437 ± 0.005 | 0.419 ± 0.009 | 0.428 ± 0.001 |
| LRd_sel | 0.438 ± 0.005 | 0.435 ± 0.002 | 0.442 ± 0.002 |
| LC_sel | 0.441 ± 0.002 | 0.427 ± 0.006 | 0.426 ± 0.002 |
| CLOCs-R_sel | 0.448 ± 0.003 | 0.443 ± 0.008 | 0.427 ± 0.002 |
| B_sel | 0.449 ± 0.002 | 0.440 ± 0.003 | 0.443 ± 0.002 |

B_sel − B and B_sel − CLOCs-R_sel agree at 50–200 m only after rounding. The unrounded values are −0.00236 and
−0.00238, and the recording intervals differ.

## Training time

Times are wall-clock on one shared GPU, with three seeds training at a time as in the paper's runs. They come from
`runs/select_trained/times.tsv` and `run.log`, in EST, on 2026-11-26.

| Step | Wall time |
|---|---|
| Step 0 (fit-trained CLOCs-R seed 1, retrained and scored on val) | 15.3 min (13:04–13:19) |
| Equivalence check | 0.8 min |
| Primary networks, 4 models × 6 seeds × 53,979 steps | 83.5 min (about 10–11 min per batch of 3 seeds) |
| Primary B_sel, 6 seeds (3 × 2 out-of-fold experts + blend) | 53.6 min (about 26 min per seed) |
| Secondary networks, 4 models × 6 seeds × 10,227 steps | 46.5 min |
| Secondary B_sel, 6 seeds | 13.3 min |
| **All training on `select`** | **197.0 min (13:20–16:37)** |
| Val scoring: 91 model-seeds, with both bootstrap levels | 205.1 min (16:37–20:02) |

Step counts:
- The primary networks each made 53,979 optimizer steps, which is 3 epochs × 17,993 `fit` frames. On `select`'s
  3,409 frames that is 15.83 passes.
- The secondary networks each made 10,227 steps, which is 3 × 3,409.
- The training logs show every primary run's schedule as `/53979`, and every secondary run's as `/10227`.

Disk: 4.7 GB free at the start and at the end.

## Deviations and choices the prompt did not fix

1. **The order of the checks.** `run.sh` ran Step 0 and the equivalence check first and then stopped (`CHECKS_ONLY=1`).
   Neither check trains anything on `select`. PREDICTIONS.md, the code and `step0.json` were then committed (f4b3232),
   and the full run started 20 s later.
   - The commit was made by hand, outside the automated session, because the session's permission settings refused
     the commit command.
2. **Code changes.** The changes were minimal, and their defaults leave every existing run unchanged.
   - `--train_split` already existed.
   - Added the **NC** token for L0. NC gives every candidate only the reserved cell `[-1, -1, s3D, d]`, ported from the
     thesis branch, whose saved L0 networks load unchanged.
   - Added **`--steps`**, in `tune_fusion.py` and `train_fusion.fit`.
   - B's blend was ported from the thesis branch to `results/select_trained/blend.py`.
   - `docs/MODELS.md` lists the new token.
   - No existing default was changed.
3. **How the step limit works.** With `--steps`, training draws fresh frame permutations until the step limit and cuts
   the last one off. The one-cycle schedule spans exactly that many steps.
4. **The recorded step count.** The per-seed `steps` field in `runs/select_trained/train_*.json` holds the last
   *logged* step, and the loss is logged every 200 steps. The field therefore reads 53,800 and 10,200. The updates
   actually made are 53,979 and 10,227:
   - the loop stops at `step == steps`;
   - the cut-off check made exactly the requested 300 updates;
   - the logs print `/53979` and `/10227`.
5. **B_sel folds.** The 3 recording folds come from `fit`'s rule (`splits._fold_of`) applied to `select`, and give
   1,845, 994 and 570 frames. They are uneven because `select` has 1–4 recordings per condition, and they were kept as
   the rule made them.
6. **Training length of B_sel's out-of-fold experts.** The prompt does not fix it. They make as many passes as the
   full experts (15.83):
   - primary: round(53,979 × frames / 3,409), which gives 24,765, 38,240 and 44,953 steps;
   - secondary: 3 epochs, which gives 4,692, 7,245 and 8,517 steps.
   The blend itself trains for 53,979 steps (primary) or 10,227 (secondary). Its start (2, 0, 0), loss and schedule
   are the thesis `train_gate`'s.
7. **Fit-trained counterparts.** These were scored from the paper's saved networks rather than read from the stored
   files, so the bootstraps could be seed-paired in the same pass. They reproduce the stored values; see the checks
   above.
8. **Winning-cell shares.**
   - They are reported as the mean of the six per-seed shares, with seed 1 alongside.
   - The bootstrap recomputes each seed's share from the resampled counts and then averages over seeds.
   - Shares are given in percent with one decimal, which is 3 decimals as a fraction.
   - The reserved-cell share is identical in all networks (2.5% and 29.9%), and so is its difference interval of
     [0, 0]. These are the candidates with no camera or radar cell, for which the reserved cell is the only one.
9. **Extra numbers.**
   - The 50–100 m differences were also computed (`results.json`, `differences`).
   - The by-condition table also shows the fit-trained models.
10. **In-sample scores.** Each select-trained network was also scored on `select` itself. These scores are recorded in
    `runs/select_trained/train_*.json`; they are in-sample, so nothing uses them.
11. **Parallel val scoring.** Val scoring ran in 5 processes at once. The numbers do not depend on this.

Nothing else went differently from `PREDICTIONS.md`. No setting was tuned, no val number of a select-trained model
was looked at before all training had finished, and nothing outside `results/select_trained/` was written in
`results/`.

## Files

- `PREDICTIONS.md`: the predictions, fixed before training.
- `step0.json` and `equivalence.json`: the two checks.
- `val.json`: every val number, from `score_val.py`.
- `results.json`: verdicts, tables, blend weights, training time and reproduction, from `make_report.py`.
- `report_tables.md`: the tables above, as generated.
- The code: `run.sh`, `blend.py`, `check_equivalence.py`, `score_val.py` and `make_report.py`.
- The networks and logs are in `runs/select_trained/`. They are not tracked.
