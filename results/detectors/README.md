# Detector baselines: SECOND and Cascade R-CNN on aiMotive and KITTI

Each detector scored alone, with no fusion and no radar. These are the frozen inputs to CLOCs-R. Run on 2026-11-01.

## Main results

- **At lenient matching, SECOND on aiMotive is level with SECOND on KITTI inside KITTI's range**, and it reaches ranges KITTI cannot: BEV AP@0.3 is 0.975 vs 0.973 at 0-30 m and 0.847 vs 0.867 at 30-50 m.
- **At strict matching it is far lower** (AP3D@0.7 0.066 vs 0.704 overall). A large part of this is a label difference between aiMotive's train and val sets, not only detector error (see [Label differences](#label-differences-between-train-and-val)). APBEV@0.3 and center-distance AP are the fairer measures on this dataset.
- **Both detectors weaken most at 100-200 m**, the range radar is meant to help: SECOND has BEV AP@0.3 of 0.277, and Cascade R-CNN finds 32% of cars at IoU 0.5.

The two datasets differ in sensors, labels and range, so the KITTI numbers are context for the pipeline, not a head-to-head comparison.

## Models

| | SECOND, aiMotive | Cascade R-CNN, aiMotive | SECOND, KITTI | Cascade R-CNN, KITTI |
|---|---|---|---|---|
| Source | Trained here | Trained here | CLOCs authors' weights (`voxelnet-30950`) | CLOCs authors' saved detections; no weights available |
| Start | Random weights | COCO-pretrained | | |
| Training | 40 epochs, about 2 h 50 min | 4 epochs, about 1 h 10 min | | |
| Frames | 7,753 (109 scenes) | 7,753 (109 scenes) | | |
| Final training loss | 0.30 | 0.29 | | |
| Config | [`second_aimotive_config.yaml`](second_aimotive_config.yaml) | [`cascade_rcnn_aimotive_config.py`](cascade_rcnn_aimotive_config.py) | `second/configs/car.fhd.config` | |

aiMotive data: daytime clear weather (highway and urban), every 2nd training frame, seed 3. Weights are in `pretrained/` (not in git, because Cascade R-CNN's 277 MB is over GitHub's 100 MB file limit), with hashes in `pretrained/SHA256SUMS`. To share them, publish them as a GitHub Release:

```bash
gh release create detectors-v1 pretrained/*.pth pretrained/SHA256SUMS --repo Richard091503/CLOCs-R --title "aiMotive detector weights" --notes "SECOND and Cascade R-CNN trained on aiMotive (CC BY-NC-SA 4.0)."
```

The repository is public, so a release will be too. Trained weights are derived from aiMotive, so they fall under its non-commercial, share-alike license.

## SECOND

Same evaluator on both datasets (`clocs_r/metrics.py`): no score threshold, BEV NMS at IoU 0.1. Car class. Trucks, buses and trailers are ignored on aiMotive; vans, trucks and trams are ignored on KITTI.

| Distance | n GT | AP3D@0.5 | AP3D@0.7 | APBEV@0.3 | APCD | Recall@BEV0.3 |
|---|---|---|---|---|---|---|
| **aiMotive val** (4,292 frames) | | | | | | |
| All | 17,669 | 0.336 | 0.066 | 0.630 | 0.556 | 0.807 |
| 0-30 m | 2,474 | 0.950 | 0.393 | 0.975 | 0.909 | 0.987 |
| 30-50 m | 1,470 | 0.514 | 0.041 | 0.847 | 0.664 | 0.920 |
| 50-100 m | 6,908 | 0.323 | 0.024 | 0.725 | 0.615 | 0.858 |
| 100-200 m | 6,777 | 0.052 | 0.001 | 0.277 | 0.280 | 0.650 |
| **KITTI val** (3,769 frames) | | | | | | |
| All | 14,219 | 0.904 | 0.704 | 0.929 | 0.925 | 0.976 |
| 0-30 m | 8,714 | 0.966 | 0.854 | 0.973 | 0.971 | 0.986 |
| 30-50 m | 4,256 | 0.830 | 0.496 | 0.867 | 0.861 | 0.959 |
| 50-70.4 m | 1,249 | 0.409 | 0.104 | 0.481 | 0.467 | 0.911 |

KITTI SECOND's grid ends at 70.4 m, so it has no results beyond that.

**Official KITTI metric, for checking against the literature** (Car, IoU 0.7, Easy / Moderate / Hard, [`kitti_second_official.txt`](kitti_second_official.txt)): 3D AP 90.82 / 79.73 / 76.87. The CLOCs README gives 90.97 / 79.94 / 77.09. This metric cannot be computed on aiMotive, which has no KITTI difficulty levels.

## Cascade R-CNN

A 2D box has no distance, so AP is binned by box height in pixels. Recall is binned by the distance of the 3D label, and uses all candidates, which is what fusion receives.

| AP | aiMotive val AP2D@0.5 | aiMotive val AP2D@0.7 | KITTI val AP2D@0.5 | KITTI val AP2D@0.7 |
|---|---|---|---|---|
| All | 0.431 | 0.229 | 0.827 | 0.739 |
| Height at least 40 px | 0.827 | 0.702 | 0.812 | 0.737 |
| 25-40 px | 0.341 | 0.111 | 0.719 | 0.653 |
| Under 25 px | 0.149 | 0.028 | 0.464 | 0.347 |

| Candidate recall | aiMotive val @0.3 | aiMotive val @0.5 | KITTI val @0.3 | KITTI val @0.5 |
|---|---|---|---|---|
| 0-30 m | 1.000 | 0.998 | 0.982 | 0.948 |
| 30-50 m | 0.982 | 0.955 | 0.992 | 0.964 |
| 50-100 m | 0.793 | 0.702 | 0.870 | 0.802 |
| 100-200 m | 0.492 | 0.320 | no labels | no labels |

Three differences limit this comparison:

- **Far-car share:** 57% of aiMotive's labeled cars are under 25 px tall, against 12% on KITTI, which drives its lower overall AP.
- **Labels:** aiMotive's 2D labels are projected 3D boxes, which include occluded cars. KITTI's are hand-drawn.
- **Candidates:** aiMotive keeps up to 200 per frame (median 200), and the authors' KITTI files have a median of 39.

## By driving condition

| | SECOND APBEV@0.3, all | SECOND APBEV@0.3, 100-200 m | Cascade AP2D@0.5 | Cascade recall@0.5, 100-200 m |
|---|---|---|---|---|
| **Val** | | | | |
| Overall | 0.630 | 0.277 | 0.431 | 0.320 |
| Highway | 0.639 | 0.318 | 0.442 | 0.317 |
| Urban | 0.633 | 0.270 | 0.433 | 0.321 |
| Without tunnel scenes | 0.634 | 0.274 | 0.436 | 0.330 |
| **Holdout** (923 frames) | | | | |
| Overall | 0.629 | 0.313 | 0.499 | 0.307 |
| Highway | 0.930 | 0.656 | 0.810 | 0.734 |
| Urban | 0.520 | 0.001 | 0.403 | 0.063 |
| Without tunnel scenes | 0.618 | 0.288 | 0.489 | 0.287 |

Five val scenes and one holdout scene contain tunnels (`splits.TUNNEL_SCENES`), where the camera is badly underexposed, and no training scene does. Removing them changes the scores very little. The holdout set has only 466 cars at 100-200 m in the SECOND evaluation (415 of Cascade's 652 are urban), so its far-range figures are noisy.

For reference, the aiMotive paper's own LiDAR baseline reports BEV AP@0.3 of 0.755 on highway and 0.608 on urban ([Table 6](https://arxiv.org/abs/2211.09445)). It uses a different model, all object classes, Hungarian matching, and a region extending behind the car as well, so these figures only indicate the expected range.

## Label differences between train and val

The aiMotive paper states that training scenes are auto-annotated with manual quality checks, and that the validation set was labeled by hired annotators, and warns that this may cause some distribution shift. The car labels confirm it:

| Car labels | Length, median (p10-p90), m | Width, m | Height, m | Exactly 4.00 m long |
|---|---|---|---|---|
| Train | 4.00 (3.82-5.02) | 2.02 (1.00-2.57) | 1.47 (0.95-1.92) | 43% |
| Holdout | 4.00 (4.00-4.87) | 2.04 (0.91-2.54) | 1.46 (0.86-1.98) | 58% |
| Val | 4.37 (3.85-5.76) | 1.91 (1.75-2.28) | 1.68 (1.46-2.22) | 0% |

SECOND learned the training convention. Median box errors (detection minus label) are close to zero on holdout, but systematic on val:

| Val, cars matched at BEV IoU 0.3 | Center, x | Length | Width | Height | Height position | Median BEV IoU |
|---|---|---|---|---|---|---|
| 0-30 m | -0.22 m | -0.04 m | +0.20 m | -0.06 m | 0.00 m | 0.78 |
| 30-50 m | -0.38 m | -0.13 m | +0.25 m | -0.10 m | -0.01 m | 0.68 |
| 50-100 m | -0.20 m | -0.10 m | +0.21 m | -0.13 m | 0.00 m | 0.71 |
| 100-200 m | -0.28 m | -0.42 m | +0.05 m | -0.24 m | -0.02 m | 0.68 |

This is why AP falls so quickly with a stricter threshold on val: BEV AP is 0.699 @0.1, 0.630 @0.3, 0.484 @0.5 and 0.207 @0.7. CLOCs only rescores boxes and does not move them, so fusion cannot fix this.

The detectors were not adjusted after seeing val, and should stay that way so val remains a clean test set.

## Generalization to unseen scenes

| | 900 training frames | Holdout |
|---|---|---|
| SECOND AP3D@0.7, 0-30 m | 0.815 | 0.485 |
| SECOND APBEV@0.3, all | 0.835 | 0.629 |
| Cascade AP2D@0.5, all | 0.803 | 0.499 |
| Cascade recall@0.5, 100-200 m | 0.835 | 0.307 |

Both detectors score much higher on frames they were trained on. Most of the holdout loss is in urban scenes. We have not separated memorized scenes from label noise as causes: 10% of training car labels are 1.00 m wide or less, and SECOND kept only its last two checkpoints, so earlier epochs cannot be compared.

At long range, LiDAR coverage is the main limit. On holdout, 49% of cars at 100-200 m have no LiDAR points and 66% have fewer than 5. Counting only cars with at least 5 points raises SECOND's APBEV@0.3 there from 0.313 to 0.817 (158 cars).

## Notes for the paper

- **Metrics:** lead with APBEV@0.3 and center-distance AP on aiMotive, and still report AP3D with the label-shift explanation above.
- **Detectors stay frozen:** the decision not to retrain was made on holdout, before val was run. Both detectors have now been scored on val, so any retrain from here would have to be disclosed.
- **KITTI Cascade R-CNN:** these numbers come from the CLOCs authors' saved detections, because their weights are not available.
- **Small far-range selection set:** holdout has 466 cars at 100-200 m in the SECOND evaluation, against 6,777 in val, so choices made on holdout at that range rest on a small sample.
- **Fusion:** CLOCs without radar has now been scored against these detectors in [`../clocs_no_radar/`](../clocs_no_radar/README.md). Both detectors look much better on training frames than on new scenes, but training fusion on unseen scenes did not improve it.

## Reproducing

```bash
source env.sh
python -m clocs_r.export_3d --ckpt pretrained/second_aimotive.pth --out runs/candidates3d --splits holdout,val
python -m clocs_r.cascade_2d export --out runs/cascade --ckpt pretrained/cascade_rcnn_aimotive.pth --candidates runs/candidates2d --splits holdout,val
python -m clocs_r.eval_detectors --splits holdout,val --out results/detectors
(cd second && python -m clocs_r.eval_kitti --out ../results/detectors)
```

Full numbers are in [`aimotive_second.json`](aimotive_second.json), [`aimotive_cascade.json`](aimotive_cascade.json), [`kitti_second.json`](kitti_second.json) and [`kitti_cascade.json`](kitti_cascade.json). The label, box-error and train-versus-holdout tables came from one-off checks. [`diag_second.py`](diag_second.py) and [`diag_second.json`](diag_second.json) are the saved SECOND part of those.
