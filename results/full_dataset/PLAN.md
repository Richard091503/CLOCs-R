# Full-dataset study plan: CLOCs vs CLOCs-R in adverse conditions, focused on range

Written on 2026-11-14, before any detector or fusion result on the full dataset was seen.

## Question

How much does radar add to camera-LiDAR CLOCs at long range (beyond 50 m), and is that contribution larger at night and in rain than in clear daytime weather?

## Data

The whole aiMotive dataset: all four driving conditions (highway, urban, night, rain) and every annotated frame (26,583). Earlier results in `results/` used daytime highway and urban only, with every 2nd training frame.

| Split | Frames | Scenes | Highway | Urban | Night | Rain |
|---|---|---|---|---|---|---|
| train (detectors and fusion) | 18,994 | 134 | 8,924 | 6,525 | 3,117 | 428 |
| holdout (selection only) | 2,408 | 17 | 990 | 851 | 425 | 142 |
| val (reported once) | 5,181 | 25 | 2,252 | 2,040 | 444 | 445 |

Holdout is every 10th training scene per condition, as before (`clocs_r/splits.py`). Night and rain val have only 2 scenes each, so condition-level differences there are reported with that limit stated.

## Range focus (fixed before training)

- **SECOND:** classification, box and direction losses weighted by anchor range, x1 below 40 m, rising linearly to x2 at 60 m and beyond (`clocs_r/range_weight.py`, `clocs_r/second_head.py`). The training voxel cap goes from 16,000 to 20,000; measurement showed this hardly mattered (0.7% of frames above 16,000, 0.03% of voxels lost). 40 epochs, as before.
- **Cascade R-CNN:** images upscaled 1.5x (1920x1056) for training and inference, for small far cars. No loss weighting, since a 2D box has no range of its own in mmdet's losses. 4 epochs, as before.
- **Fusion:** the same range weighting on the focal loss, by candidate range (`--far_weight 2`).

No plain detectors are trained, so the effect of range weighting on the detectors is not measured separately. The fusion-level effect is measured by one extra run (see below).

## Fusion configurations

All use the detectors above, the existing gate (delta0 = 0.5 m, kappa = 0.02; not reselected), 3 sweeps, 3 epochs, seeds 1, 2 and 3, and far weight 2 unless stated.

| Name | Radar | Residual (T2) | Role |
|---|---|---|---|
| R0 | none | no | CLOCs (camera-LiDAR) |
| R0+T2 | none | yes | CLOCs with the residual fix |
| R2 | Doppler speed | no | CLOCs-R candidate |
| R3 | support + Doppler speed | no | CLOCs-R candidate (the paper's proposal) |
| R2+T2 | Doppler speed | yes | CLOCs-R candidate (earlier tuned winner) |
| R3+T2 | support + Doppler speed | yes | CLOCs-R candidate |
| R0, far weight 1 | none | no | CLOCs with the original loss, for the effect of range weighting on fusion |

## Selection rule

1. Score the four CLOCs-R candidates on holdout by center-distance AP beyond 50 m (`overall`, bin `50-200`), mean of 3 seeds.
2. The highest mean wins. Val is not read until this is done.
3. Its radar-free twin (R0 if the winner has no T2, R0+T2 if it has) is the comparison for "what radar adds".

## Reporting on val (once, after selection)

- SECOND and Cascade R-CNN alone, every fusion configuration above, with the selected CLOCs-R marked; the others are context.
- Metrics: center-distance AP and BEV AP@0.3 first (label differences between aiMotive train and val limit AP3D, see `results/detectors/README.md`), AP3D@0.5/0.7 and recall as well.
- Distance bins 0-30, 30-50, 50-100, 100-200 and beyond 50 m; groups overall, clear (highway + urban), adverse (night + rain), each condition, and without tunnel scenes.
- **Radar gain** = selected CLOCs-R minus its twin, per group and bin, as the seed mean with the seed range, plus per-scene differences for the adverse scenes.
- Holdout numbers are shown beside val. Holdout was used for selection, so it is the optimistic one.

## Addendum, 2026-11-15 (after the val results above were seen)

At the user's request, the reported CLOCs-R model was switched to R3+T2, the configuration that uses both radar features (number of radar points and Doppler speed). R3+T2, its twin R0+T2 and, for a like-for-like comparison, R2+T2 were trained with 3 further seeds (4, 5, 6) and pooled with seeds 1-3. This is a post-hoc choice, not the outcome of the selection rule above: R2+T2 remains the selected model, and its tables are kept unchanged in `tables.md`. The R3+T2 report is in `tables_r3.md`.
