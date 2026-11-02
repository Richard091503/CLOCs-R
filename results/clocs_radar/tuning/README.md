# Fusion tuning

Six changes to the fusion stage, tested on holdout under a [plan](PLAN.md) written before any tuning run. **This tuning is post hoc:** val had already been seen for the as-designed models, so these results are reported alongside them and do not replace them. Run on 2026-11-01. Detectors, gate and network size were not changed.

## Main results

- **One change works: making fusion adjust SECOND's score rather than replace it (T2).** It raises center-distance AP beyond 50 m on val from 0.428 to 0.451, and fixes most of the loss to SECOND.
- **Most of that gain comes from repairing fusion, not from radar.** The same change without radar reaches 0.448, so radar contributes about 0.003 there, and the tuned model beats it in 10 of 20 scenes.
- **The tuned model now matches or beats SECOND on BEV AP and recall, but not on center-distance AP at range.** Beyond 50 m on val, it leads on BEV AP@0.3 (0.524 vs 0.517), while SECOND still leads on center-distance AP (0.458 vs 0.451), the paper's selection metric. SECOND also leads on center-distance AP overall and at 50-100 m.
- **Radar's clearest contribution remains at 30-50 m,** where BEV AP@0.3 goes from 0.861 to 0.874 and the seed ranges do not overlap.
- **Getting more radar returns did not help.** 10 sweeps with motion correction reach more far cars (70% vs 58% on holdout) but do not raise the score, and neither does RCS.

## Selection on holdout

Center-distance AP beyond 50 m, mean of 3 seeds. A change had to reach 0.4582 to be adopted, as fixed in the plan. SECOND scores 0.4601.

| Change | Holdout | Seeds | Adopted |
|---|---|---|---|
| R2, unchanged | 0.4545 | 0.4512, 0.4536, 0.4588 | Baseline (0.4554 in the ablation) |
| T1 far targets | 0.4433 | 0.4430, 0.4439, 0.4431 | No |
| **T2 residual** | **0.4619** | 0.4601, 0.4628, 0.4629 | **Yes** |
| T3 logit input | 0.4502 | 0.4508, 0.4467, 0.4530 | No |
| T4a 5 sweeps with motion correction | 0.4552 | 0.4560, 0.4520, 0.4577 | No |
| T4b 10 sweeps with motion correction | 0.4550 | 0.4553, 0.4524, 0.4572 | No |
| T5 RCS | 0.4529 | 0.4522, 0.4524, 0.4541 | No |

Only T2 was adopted, so there was no combination to test, and T2 won under the rule.

T2 is simple: the fused logit is SECOND's logit plus the network's output, and the last layer starts at zero, so training begins exactly at SECOND's ranking. It adds no parameters.

## Final comparison on val

The winner was scored on val after selection, as the plan allows. R2 + T2 is the tuned model, and R0 + T2 is the same change without radar. Means of 3 seeds, with the range in brackets.

| APBEV@0.3 | SECOND | R0 | R2 | R0 + T2 | R2 + T2 |
|---|---|---|---|---|---|
| All | 0.630 | 0.599 | 0.607 | 0.623 (0.618-0.627) | **0.630** (0.627-0.633) |
| 0-30 m | 0.975 | 0.974 | 0.974 | 0.974 (0.974-0.974) | 0.974 (0.973-0.974) |
| 30-50 m | 0.847 | 0.857 | 0.870 | 0.861 (0.859-0.862) | **0.874** (0.873-0.875) |
| 50-100 m | 0.725 | 0.701 | 0.706 | 0.723 (0.720-0.726) | **0.727** (0.725-0.730) |
| 100-200 m | 0.277 | 0.267 | 0.271 | 0.296 (0.289-0.305) | **0.298** (0.295-0.303) |
| Beyond 50 m | 0.517 | 0.493 | 0.497 | 0.521 (0.516-0.525) | **0.524** (0.520-0.526) |

| Center-distance AP | SECOND | R0 | R2 | R0 + T2 | R2 + T2 |
|---|---|---|---|---|---|
| All | **0.556** | 0.525 | 0.530 | 0.545 (0.541-0.549) | 0.551 (0.546-0.554) |
| 0-30 m | 0.909 | **0.926** | 0.918 | 0.925 (0.924-0.925) | 0.922 (0.921-0.923) |
| 30-50 m | 0.664 | 0.693 | 0.698 | 0.697 (0.696-0.698) | **0.704** (0.702-0.706) |
| 50-100 m | **0.615** | 0.589 | 0.593 | 0.606 (0.604-0.608) | 0.609 (0.608-0.611) |
| 100-200 m | 0.280 | 0.252 | 0.255 | 0.280 (0.272-0.288) | 0.283 (0.276-0.289) |
| **Beyond 50 m** | **0.458** | 0.424 | 0.428 | 0.448 (0.444-0.453) | 0.451 (0.447-0.454) |

| Other val metrics | SECOND | R0 | R2 | R0 + T2 | R2 + T2 |
|---|---|---|---|---|---|
| AP3D@0.5, all | 0.336 | 0.349 | 0.350 | 0.354 | 0.354 |
| AP3D@0.5, 30-50 m | 0.514 | 0.603 | 0.595 | 0.602 | 0.599 |
| AP3D@0.7, all | 0.066 | 0.063 | 0.062 | 0.066 | 0.066 |
| AP3D@0.7, 0-30 m | 0.393 | 0.413 | 0.381 | 0.416 | 0.405 |
| Recall, all | 0.807 | 0.794 | 0.798 | 0.815 | 0.817 |
| Recall, 100-200 m | 0.650 | 0.614 | 0.623 | 0.655 | 0.659 |

Recall is at BEV IoU 0.3. T2 also recovers the side effect at strict close-range matching: AP3D@0.7 at 0-30 m returns from 0.381 to 0.405, though R0 + T2 is still higher at 0.416.

### Holdout, for comparison

| Center-distance AP | SECOND | R0 | R2 | R0 + T2 | R2 + T2 |
|---|---|---|---|---|---|
| All | 0.622 | 0.615 | 0.619 | 0.622 | 0.626 |
| 30-50 m | 0.670 | 0.669 | 0.674 | 0.674 | 0.680 |
| 50-100 m | 0.478 | 0.465 | 0.470 | 0.475 | 0.479 |
| 100-200 m | 0.379 | 0.384 | 0.389 | 0.384 | 0.388 |
| Beyond 50 m | 0.460 | 0.450 | 0.455 | 0.458 | 0.462 |

On holdout, R2 + T2 passes SECOND beyond 50 m, and on val it does not, so holdout is an imperfect guide to val. The difference between auto-generated and hand-made labels may be part of the reason, but we did not test that.

### Val by driving condition

| APBEV@0.3 / center-distance AP | SECOND | R0 + T2 | R2 + T2 |
|---|---|---|---|
| Highway, all | 0.639 / 0.559 | 0.635 / 0.542 | 0.639 / 0.545 |
| Highway, 100-200 m | 0.318 / 0.299 | 0.333 / 0.282 | 0.340 / 0.291 |
| Urban, all | 0.633 / 0.562 | 0.623 / 0.551 | 0.632 / 0.558 |
| Urban, 100-200 m | 0.270 / 0.284 | 0.291 / 0.288 | 0.291 / 0.289 |
| Without tunnel scenes, all | 0.634 / 0.558 | 0.628 / 0.551 | 0.637 / 0.557 |
| Without tunnel scenes, 100-200 m | 0.274 / 0.273 | 0.299 / 0.286 | 0.300 / 0.288 |

## By scene, on val

Center-distance AP beyond 50 m, for the 20 scenes with at least 50 cars beyond 50 m, using seed means ([`compare_scenes.json`](compare_scenes.json)):

- R2 + T2 beats R0 + T2 in 10 scenes, with a median difference of about zero.
- R2 + T2 beats SECOND in 10 scenes, with a median difference of about zero.

## What this means

- **The radar question:** radar's contribution beyond 50 m stays small after tuning, about 0.003 on center-distance AP and within seed spread. A better fusion design gained about 0.02 there. So the shortfall we saw came mainly from how CLOCs replaced SECOND's score, and radar did not close it.
- **Why T2 works is supported, not proven.** The earlier analysis showed fusion ranking far boxes worse than SECOND. Starting from SECOND's ranking and learning a correction fits that, but we did not test the mechanism separately.
- **The camera finding should be read again:** the [ablation](../ablation/README.md#camera-removed) showed removing the camera raising the far score. With T2, R0 reaches 0.448 with the camera in place, about the same as removing it (0.444).

## Limits on these numbers

- **Val is no longer a fully untouched test set, and the paper should say so.** The choice among these changes used holdout only, but the list of changes came partly from analysis of the earlier val results, including the finding that fusion ranks far boxes worse than SECOND. A clean estimate of T2 would need new data.
- Holdout has only 466 cars at 100-200 m and auto-generated labels.
- Seven configurations were compared on holdout, so the winner's score there is slightly optimistic. The val score is the one to report.
- The T4 results use a fixed gate. Selecting the gate again for more sweeps was outside the plan.

## Files

- [`PLAN.md`](PLAN.md): the changes and the rule, written before the runs
- `selection.json`: all changes on holdout, by seed, bin and condition
- `final.json`: R2 + T2 and R0 + T2 on holdout and val
- `T2_seed{1,2,3}.pth`, `R0_T2_seed{1,2,3}.pth`: weights, which load with `fusion.FusionNet(m, residual=True)`
- `compare_scenes.py/json`: the scene comparison

## Reproducing

```bash
source env.sh
python -m clocs_r.tune_fusion --configs R2,T1,T2,T3,T4a,T4b,T5 --out results/clocs_radar/tuning/selection.json
python -m clocs_r.tune_fusion --configs T2,R0+T2 --splits holdout,val --out results/clocs_radar/tuning/final.json --weights runs/tuning/final
python results/clocs_radar/tuning/compare_scenes.py
```
