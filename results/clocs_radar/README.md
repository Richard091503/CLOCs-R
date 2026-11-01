# CLOCs-R: adding radar on aiMotive

Radar added to CLOCs using feature set R3 (support and Doppler speed), the pair the paper proposes, compared with SECOND alone and CLOCs without radar (R0). Run on 2026-11-01. The other feature sets and the camera-removed test belong to the ablation, which has not been run.

## Main results

- **Radar does not deliver the long-range gain we expected.** On the paper's selection metric, center-distance AP beyond 50 m on val, SECOND scores 0.458, R0 0.424 and R3 0.426. R3 beats R0 in 10 of 20 scenes and SECOND in 5 of 20.
- **There is one clear gain, at 30-50 m:** BEV AP@0.3 rises from 0.857 (R0) to 0.867, and the seed ranges do not overlap. SECOND alone scores 0.847.
- **There are small losses at strict matching at close range:** AP3D@0.7 at 0-30 m falls from 0.413 to 0.384, and holdout falls the same way.
- **The likely causes are weak radar evidence at range and a network that learned to ignore it.** At 100-200 m on val, 62% of real cars have a return, but so do 39% of false detections, and the trained network's score barely changes with radar. These explanations are supported by the measurements below, not proven.

This agrees with earlier work cited in our paper: the aiMotive baselines barely changed overall AP beyond 75 m when radar was added to LiDAR, and RadarNet gained only 0.8 AP at 70-100 m.

## Comparison

R0 and R3 are means of 3 seeds, with the range in brackets. Same evaluator and same candidates throughout.

| aiMotive val (4,292 frames) | n GT | APBEV@0.3 SECOND | APBEV@0.3 R0 | APBEV@0.3 R3 | APCD SECOND | APCD R0 | APCD R3 |
|---|---|---|---|---|---|---|---|
| All | 17,669 | 0.630 | 0.599 (0.593-0.604) | 0.607 (0.599-0.612) | 0.556 | 0.525 (0.521-0.528) | 0.529 (0.522-0.533) |
| 0-30 m | 2,474 | 0.975 | 0.974 (0.973-0.974) | 0.974 (0.973-0.974) | 0.909 | 0.926 (0.924-0.926) | 0.918 (0.915-0.919) |
| 30-50 m | 1,470 | 0.847 | 0.857 (0.856-0.859) | 0.867 (0.866-0.868) | 0.664 | 0.693 (0.693-0.695) | 0.695 (0.693-0.699) |
| 50-100 m | 6,908 | 0.725 | 0.701 (0.697-0.707) | 0.703 (0.698-0.706) | 0.615 | 0.589 (0.585-0.594) | 0.589 (0.586-0.592) |
| 100-200 m | 6,777 | 0.277 | 0.267 (0.253-0.279) | 0.272 (0.255-0.286) | 0.280 | 0.252 (0.241-0.261) | 0.257 (0.240-0.272) |
| **Beyond 50 m** | 13,685 | 0.517 | 0.493 (0.485-0.501) | 0.495 (0.485-0.502) | 0.458 | 0.424 (0.418-0.430) | 0.426 (0.417-0.432) |

| aiMotive val | AP3D@0.5 SECOND | R0 | R3 | AP3D@0.7 SECOND | R0 | R3 | Recall SECOND | R0 | R3 |
|---|---|---|---|---|---|---|---|---|---|
| All | 0.336 | 0.349 | 0.346 | 0.066 | 0.063 | 0.061 | 0.807 | 0.794 | 0.798 |
| 0-30 m | 0.950 | 0.958 | 0.957 | 0.393 | 0.413 | 0.384 | 0.987 | 0.987 | 0.987 |
| 30-50 m | 0.514 | 0.603 | 0.587 | 0.041 | 0.047 | 0.042 | 0.920 | 0.924 | 0.924 |
| 50-100 m | 0.323 | 0.328 | 0.323 | 0.024 | 0.018 | 0.018 | 0.858 | 0.858 | 0.861 |
| 100-200 m | 0.052 | 0.047 | 0.047 | 0.001 | 0.001 | 0.001 | 0.650 | 0.614 | 0.622 |

Recall is at BEV IoU 0.3. AP3D@0.7 is limited by the label difference described in [the detector results](../detectors/README.md#label-differences-between-train-and-val).

The KITTI columns are in [`../clocs_no_radar/`](../clocs_no_radar/README.md). KITTI has no radar, so R3 cannot be run on it.

### By driving condition

| | APBEV@0.3, all: SECOND | R0 | R3 | APBEV@0.3, 100-200 m: SECOND | R0 | R3 |
|---|---|---|---|---|---|---|
| **Val** | | | | | | |
| Overall | 0.630 | 0.599 | 0.607 | 0.277 | 0.267 | 0.272 |
| Highway | 0.639 | 0.594 | 0.599 | 0.318 | 0.268 | 0.277 |
| Urban | 0.633 | 0.603 | 0.612 | 0.270 | 0.274 | 0.277 |
| Without tunnel scenes | 0.634 | 0.608 | 0.616 | 0.274 | 0.274 | 0.279 |
| **Holdout** | | | | | | |
| Overall | 0.629 | 0.620 | 0.623 | 0.313 | 0.319 | 0.325 |
| Highway | 0.930 | 0.924 | 0.927 | 0.656 | 0.660 | 0.670 |
| Urban | 0.520 | 0.514 | 0.516 | 0.001 | 0.002 | 0.002 |
| Without tunnel scenes | 0.618 | 0.609 | 0.612 | 0.288 | 0.291 | 0.297 |

On holdout, R3 is above both baselines at 100-200 m (seed ranges do not overlap with R0), but below SECOND at 50-100 m. Holdout has only 466 cars at 100-200 m, and that gain did not carry over clearly to val.

## Gate selection

The gate (δ0, κ of Eq. 2) was chosen on holdout by center-distance AP beyond 50 m, with R3 and 3 seeds per gate. The rule, highest mean wins, was fixed before the run, and val was not read. On the same metric, SECOND scores 0.460 and R0 0.450.

| δ0 (m) | κ = 0.01 | κ = 0.02 | κ = 0.04 |
|---|---|---|---|
| 0.25 | 0.4550 | 0.4532 | 0.4532 |
| 0.5 | 0.4546 | **0.4551** | 0.4522 |
| 1.0 | 0.4549 | 0.4538 | 0.4530 |

**Selected: δ0 = 0.5 m, κ = 0.02.** This is effectively a tie. The whole grid spans 0.003, about the same as the spread between seeds of one gate (0.452-0.459 for the selected gate), so results are not sensitive to the gate within this range. Each of the 9 gates beats R0 and none reaches SECOND.

## Why radar adds so little at range

The boxes are those SECOND's NMS keeps, and a real car means BEV IoU of at least 0.3 with a label, as in the [camera analysis](../clocs_no_radar/README.md#why-clocs-loses-ground-at-range-on-aimotive). AUC measures how well a score separates real cars from false detections, from 0.5 (no signal) to 1.0 (perfect). "Score change" is the change in the fused logit when radar inputs are replaced with "no return", for boxes with a return, averaged over the 3 networks.

| Val | With a return: real cars | false boxes | AUC SECOND | Support | Doppler | R0 | R3 | Score change: real | false |
|---|---|---|---|---|---|---|---|---|---|
| 0-30 m | 53% | 10% | 0.999 | 0.720 | 0.725 | 0.999 | 0.999 | +0.45 | +0.12 |
| 30-50 m | 62% | 20% | 0.988 | 0.715 | 0.754 | 0.995 | 0.995 | +0.50 | -0.11 |
| 50-100 m | 62% | 29% | 0.972 | 0.673 | 0.714 | 0.973 | 0.975 | +0.17 | -0.02 |
| 100-200 m | 62% | 39% | 0.909 | 0.615 | 0.664 | 0.873 | 0.876 | -0.02 | 0.00 |

| Holdout | With a return: real cars | false boxes | AUC SECOND | Support | Doppler | R0 | R3 | Score change: real | false |
|---|---|---|---|---|---|---|---|---|---|
| 0-30 m | 54% | 13% | 0.987 | 0.718 | 0.709 | 0.986 | 0.986 | +0.38 | +0.15 |
| 30-50 m | 56% | 17% | 0.962 | 0.690 | 0.719 | 0.966 | 0.966 | +0.37 | +0.04 |
| 50-100 m | 66% | 31% | 0.956 | 0.686 | 0.720 | 0.951 | 0.950 | +0.26 | 0.00 |
| 100-200 m | 75% | 30% | 0.945 | 0.722 | 0.798 | 0.948 | 0.945 | +0.10 | 0.00 |

What the measurements support:

1. **The radar signal is sparse, and weakest on val at range.** A frame has about 96 returns from 3 sweeps (median), and a labeled car has about 1.9 returns in its gate at 100-200 m on val. Doppler's AUC is 0.798 on holdout and 0.664 on val there.
2. **The network learns to use radar at close range and almost ignores it far away.** Its score change falls from about +0.5 to about zero on val.
3. **Training gives it little reason to use radar at range.** The detector looks almost perfect on the frames it was trained on: SECOND's AUC at 100-200 m is 0.984 on training frames, against 0.909 on val. Only 7,232 of 136,332 training positives (5%) are at 100-200 m, among 6.9 million negatives at that range.
4. **Radar could add only a little there, even to a model built for it.** The test below fits a simple logistic model on SECOND's score with and without radar, always scored on scenes it did not train on.

| AUC on unseen scenes | Fit within val: without radar | with radar | Fit on holdout, scored on val: without | with |
|---|---|---|---|---|
| 0-30 m | 0.999 | 0.999 | 0.999 | 0.999 |
| 30-50 m | 0.987 | 0.991 | 0.989 | 0.993 |
| 50-100 m | 0.971 | 0.971 | 0.972 | 0.973 |
| 100-200 m | 0.908 | 0.917 | 0.909 | 0.909 |

At 100-200 m, radar raises this model's AUC by 0.008 when fit on val itself, and by nothing when fit on holdout, which has only 215 real cars at that range. This is not a strict upper bound, because a richer model could find more. It does show that the larger shortfall lies elsewhere: both fusion networks score about 0.875 there, below SECOND's 0.909, with or without radar. Removing radar from R3 gives 0.878, so most of this gap comes from fusing the camera and SECOND's score, as [the no-radar results](../clocs_no_radar/README.md) found.

We did not try to fix this. Val has now been seen for R3, so any change to the training would have to be selected on holdout and disclosed.

## Radar data checks

Eq. 1 was checked on all 143 sequences with two tests: returns far from any labeled object should be static (below 1 m/s), and returns on a labeled car should match its labeled velocity.

| | Sequences | Off-label returns near zero: median (lowest) | Doppler minus label, median m/s |
|---|---|---|---|
| Train | 109 | 88% (19%) | 0.06 |
| Holdout | 13 | 90% (78%) | 0.08 |
| Val | 21 | 91% (81%) | 0.18 |

**No sequence fails both tests.** Five highway training sequences have fewer than 50% of off-label returns near zero, but their Doppler matches labels to within 0.12 m/s, so those returns are probably unlabeled traffic. Five sequences (2 train, 3 val) disagree with labels by 14-26 m/s, while their off-label returns are at least 90% static. Two of the val sequences contain tunnels, where wall clutter inside the gate is a plausible cause, but we did not confirm this. Full results are in [`diag_radar.json`](diag_radar.json).

| Labeled cars with a return in the starting gate (3 sweeps) | 0-30 m | 30-50 m | 50-100 m | 100-200 m |
|---|---|---|---|---|
| Holdout | 52% | 51% | 51% | 53% |
| Val | 53% | 62% | 61% | 60% |

## Setup

| | |
|---|---|
| Features | Support and Doppler speed (R3), v_max 40 m/s; 2 radar channels, 6 inputs, 2,179 parameters |
| Radar | Front long-range radar, last 3 sweeps (M = 3), moved to the current frame with ego poses, Doppler compensated by Eq. 1 |
| Gate | δ0 = 0.5 m, κ = 0.02 |
| Training | The same network, schedule and seeds as R0, 3 epochs on 7,753 frames. Final loss about 0.084, against about 0.093 for R0 |
| Fusion forward pass | 1.16 ms per frame (median) |
| Radar features | 64.8 ms per frame (median), 105.9 ms at the 95th percentile, on one CPU thread |

The radar feature time is an upper bound for R3. The code computes all five features in a Python loop over about 550 candidates with a return per frame, and has not been optimized. Accumulating sweeps takes 0.3 ms of this. The detectors are not included.

GPU training is not bit-identical between runs, so holdout numbers can differ slightly between the selection run and the final run.

## Fixes made along the way

- **Ego velocity:** 33 of 12,876 val sweeps (0.26%) had poses but no timestamps, so ego velocity fell back to zero and Doppler stayed uncompensated. They now use the sequence's frame period, and `clocs_r/test_core.py` covers this. It only affects radar, so detector and R0 results do not change.
- **`project_boxes`** moved to `clocs_r/aimotive.py`, because importing mmdet blocked GPU use in forked workers. A rebuilt frame's cells matched the cached ones exactly.
- **Discarded analyses:** an early check compared Doppler with the wrong label field (train labels list absolute velocity first), and a lookup-table version of the information test failed because its bins were too sparse. Neither is used here. The correct versions are in the scripts.

## Files

- `aimotive_r3_seed{1,2,3}.json` and `clocs_r3_aimotive_seed{1,2,3}.pth`: per-seed results and weights
- `gate_selection.json`: all gates, seeds and bins, with SECOND and R0 on holdout
- `diag_radar.py/json`: radar data checks
- `diag_radar_signal.py/json`: signal and score-change tables
- `diag_radar_limits.py/json`: training examples, information test, and selection metric by seed and scene
- `radar_latency.json`: radar feature timing

## Reproducing

```bash
source env.sh
python results/clocs_radar/diag_radar.py
python -m clocs_r.select_gate --out results/clocs_radar/gate_selection.json
python -m clocs_r.cache_cells --split val,holdout,train --delta0 0.5 --kappa 0.02
for s in 1 2 3; do python -m clocs_r.train_fusion --feature_set R3 --delta0 0.5 --kappa 0.02 --out runs/fusion/R3_s$s --seed $s --eval_splits holdout,val; done
python -m clocs_r.radar_latency --delta0 0.5 --kappa 0.02 --out results/clocs_radar/radar_latency.json
python results/clocs_radar/diag_radar_signal.py
python results/clocs_radar/diag_radar_limits.py
```
