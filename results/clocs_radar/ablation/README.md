# Ablation: what each radar feature contributes

Every feature set in Table II of the paper, trained with the same network, schedule and seeds, using the gate selected on holdout (δ0 = 0.5 m, κ = 0.02, M = 3). Run on 2026-11-01.

## Main results

- **No radar feature set has a clear effect beyond 50 m.** All eight sets fall within 0.007 of each other on val, about the same as the spread between seeds of one set.
- **R2 (Doppler speed alone) is the selected set**, under the paper's rule of highest center-distance AP beyond 50 m on holdout. It is effectively tied with R3 and R7, so this selection should not be read as Doppler being proven best.
- **More features do not help.** R7 (all five) is no better than R2 on either split.
- **Doppler appears to carry what signal there is.** R2 has the highest val score beyond 50 m, while R1 (support alone) stays level with R0. This is a small difference, within seed spread.
- **SECOND alone still beats every set beyond 50 m**, on both holdout (0.460) and val (0.458).
- **Removing the camera raises the score beyond 50 m on val** (R0 from 0.424 to 0.444, and R2 from 0.428 to 0.448), so the camera is hurting fusion at range. This test has a caveat, described below.

## Selection metric

Center-distance AP beyond 50 m, mean of 3 seeds, with the range in brackets. Holdout chose the set; val played no part.

| Set | Features | Channels | Holdout | Val |
|---|---|---|---|---|
| SECOND alone | | | 0.460 | 0.458 |
| R0 | none | 4 | 0.450 (0.448-0.452) | 0.424 (0.418-0.430) |
| R1 | r | 5 | 0.451 (0.446-0.455) | 0.424 (0.421-0.429) |
| **R2 (selected)** | v | 5 | **0.455 (0.452-0.459)** | 0.428 (0.417-0.435) |
| R3 | r, v | 6 | 0.455 (0.452-0.458) | 0.426 (0.417-0.432) |
| R4 | r, v, e | 7 | 0.454 (0.453-0.456) | 0.427 (0.424-0.431) |
| R5 | r, Δ, o | 7 | 0.449 (0.445-0.453) | 0.421 (0.414-0.425) |
| R6 | r, v, Δ | 7 | 0.454 (0.453-0.455) | 0.422 (0.420-0.426) |
| R7 | r, v, e, Δ, o | 9 | 0.455 (0.454-0.456) | 0.423 (0.416-0.426) |

## Val, by distance

Means of 3 seeds.

| Set | APBEV@0.3 all | 0-30 m | 30-50 m | 50-100 m | 100-200 m | APCD all | 0-30 m | 30-50 m | 50-100 m | 100-200 m |
|---|---|---|---|---|---|---|---|---|---|---|
| SECOND | 0.630 | 0.975 | 0.847 | 0.725 | 0.277 | 0.556 | 0.909 | 0.664 | 0.615 | 0.280 |
| R0 | 0.599 | 0.974 | 0.857 | 0.701 | 0.267 | 0.525 | 0.926 | 0.693 | 0.589 | 0.252 |
| R1 | 0.603 | 0.975 | 0.866 | 0.702 | 0.262 | 0.528 | 0.927 | 0.697 | 0.590 | 0.249 |
| R2 | 0.607 | 0.974 | 0.870 | 0.706 | 0.271 | 0.530 | 0.918 | 0.698 | 0.593 | 0.255 |
| R3 | 0.607 | 0.974 | 0.867 | 0.703 | 0.272 | 0.529 | 0.918 | 0.695 | 0.589 | 0.257 |
| R4 | 0.607 | 0.973 | 0.868 | 0.704 | 0.269 | 0.529 | 0.916 | 0.695 | 0.591 | 0.254 |
| R5 | 0.607 | 0.975 | 0.866 | 0.702 | 0.266 | 0.527 | 0.926 | 0.690 | 0.582 | 0.250 |
| R6 | 0.606 | 0.975 | 0.870 | 0.706 | 0.262 | 0.526 | 0.919 | 0.695 | 0.589 | 0.244 |
| R7 | 0.605 | 0.975 | 0.871 | 0.701 | 0.265 | 0.527 | 0.920 | 0.695 | 0.589 | 0.249 |

The clearest radar effect is at 30-50 m, where every set raises BEV AP@0.3 from 0.857 to between 0.866 and 0.871. Support alone (R1) already delivers most of it.

Strict close-range matching shows a side effect. Sets with Doppler speed lose AP3D@0.7 at 0-30 m (R0 0.413, R2 0.381, R3 0.384, R4 0.375, R6 0.386, R7 0.384), while R1 and R5, which have no Doppler speed, do not (0.417 and 0.411). The same sets lose center-distance AP at 0-30 m (0.926 for R0, 0.916-0.920 for those with Doppler speed).

Full numbers by metric, bin and driving condition are in [`summary.json`](summary.json).

## Camera removed

The selected set, with all 2D candidates removed at test time, so every candidate uses the reserved row. R0 is added to show what remains without radar as well.

| | Holdout: APCD beyond 50 m | 100-200 m | Val: APCD beyond 50 m | 50-100 m | 100-200 m | Val: APBEV@0.3, all |
|---|---|---|---|---|---|---|
| SECOND alone | 0.460 | 0.379 | 0.458 | 0.615 | 0.280 | 0.630 |
| R0 with camera | 0.450 | 0.384 | 0.424 | 0.589 | 0.252 | 0.599 |
| R0, camera removed | 0.460 | 0.368 | 0.444 | 0.614 | 0.270 | 0.615 |
| R2 with camera | 0.455 | 0.389 | 0.428 | 0.593 | 0.255 | 0.607 |
| R2, camera removed | 0.439 | 0.353 | 0.448 | 0.602 | 0.267 | 0.620 |

**Caveat:** the reserved row is close to untrained. Almost every candidate overlaps some 2D box during training (78 of about 50,000 confident candidates do not), so these scores come from inputs the network has rarely seen, and they vary more between seeds (R2 on holdout ranges from 0.422 to 0.456). This does not measure how a network trained for missing cameras would behave, for example at night. Treat it as evidence about the camera at range on these models, and not as a robustness result.

## Latency

Median per frame, measured with nothing else running, excluding the detectors. Radar features ran on one CPU thread in unoptimized Python, and the fusion pass on the GPU.

| Set | Radar features (95th percentile) | Fusion forward pass | Total | Parameters |
|---|---|---|---|---|
| R0 | none | 1.47 ms | 1.5 ms | 2,143 |
| R1 | 3.7 ms (5.8) | 1.16 ms | 4.8 ms | 2,161 |
| R2 | 17.9 ms (28.0) | 1.11 ms | 19.0 ms | 2,161 |
| R3 | 18.0 ms (28.2) | 1.15 ms | 19.2 ms | 2,179 |
| R4 | 43.6 ms (72.1) | 1.16 ms | 44.8 ms | 2,197 |
| R5 | 20.4 ms (31.3) | 1.12 ms | 21.5 ms | 2,197 |
| R6 | 30.9 ms (50.0) | 1.12 ms | 32.0 ms | 2,197 |
| R7 | 61.0 ms (99.5) | 1.15 ms | 62.1 ms | 2,233 |

Medians over a per-candidate Python loop dominate the feature time, so it could likely be reduced with vectorized code.

## Reproducing

```bash
source env.sh
for fs in R1 R2 R4 R5 R6 R7; do for s in 1 2 3; do python -m clocs_r.train_fusion --feature_set $fs --delta0 0.5 --kappa 0.02 --out runs/fusion/${fs}_s$s --seed $s --eval_splits holdout,val; done; done
python results/clocs_radar/ablation/make_tables.py
python -m clocs_r.radar_latency --delta0 0.5 --kappa 0.02 --out results/clocs_radar/ablation/latency.json
```

R0 and R3 come from [`../../clocs_no_radar/`](../../clocs_no_radar/README.md) and [`../README.md`](../README.md). Per-seed results and weights are the `R*_seed*.json` and `R*_seed*.pth` files here.
