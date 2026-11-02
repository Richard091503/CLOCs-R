# CLOCs without radar: aiMotive and KITTI

Camera-LiDAR CLOCs (feature set R0) compared with the detectors alone, on both datasets. No radar data is used. Run on 2026-11-01. Detector scores are in [`../detectors/`](../detectors/README.md).

## Main results

- **On KITTI, CLOCs beats SECOND, most of all at long range:** BEV AP@0.3 rises from 0.481 to 0.622 at 50-70 m.
- **On aiMotive, it helps at 30-50 m and hurts beyond that:** BEV AP@0.3 goes from 0.847 to 0.857 at 30-50 m, from 0.725 to 0.701 at 50-100 m, and from 0.630 to 0.599 overall.
- **The difference is not caused by our code.** Our implementation, run on KITTI, matches the authors' CLOCs to within about 0.006 on every AP metric and distance bin.
- **The likely cause is weak camera evidence at long range.** At 100-200 m on val, only 8% of real cars have a well-aligned 2D box scoring above 0.5, and CLOCs separates real cars from false detections less well than SECOND alone. This is supported by the measurements below, not proven.

Radar therefore has to be judged against both SECOND alone and this baseline, because at long range SECOND is the stronger of the two. That comparison is in [`../clocs_radar/`](../clocs_radar/README.md).

## Comparison

Same evaluator on both datasets. aiMotive CLOCs is the mean of 3 seeds, with the range in brackets.

| aiMotive val (4,292 frames) | n GT | APBEV@0.3 SECOND | APBEV@0.3 CLOCs | APCD SECOND | APCD CLOCs |
|---|---|---|---|---|---|
| All | 17,669 | 0.630 | 0.599 (0.593-0.604) | 0.556 | 0.525 (0.521-0.528) |
| 0-30 m | 2,474 | 0.975 | 0.974 (0.973-0.974) | 0.909 | 0.926 (0.924-0.926) |
| 30-50 m | 1,470 | 0.847 | 0.857 (0.856-0.859) | 0.664 | 0.693 (0.693-0.695) |
| 50-100 m | 6,908 | 0.725 | 0.701 (0.697-0.707) | 0.615 | 0.589 (0.585-0.594) |
| 100-200 m | 6,777 | 0.277 | 0.267 (0.253-0.279) | 0.280 | 0.252 (0.241-0.261) |

| aiMotive val | AP3D@0.5 SECOND | AP3D@0.5 CLOCs | AP3D@0.7 SECOND | AP3D@0.7 CLOCs | Recall SECOND | Recall CLOCs |
|---|---|---|---|---|---|---|
| All | 0.336 | 0.349 | 0.066 | 0.063 | 0.807 | 0.794 |
| 0-30 m | 0.950 | 0.958 | 0.393 | 0.413 | 0.987 | 0.987 |
| 30-50 m | 0.514 | 0.603 | 0.041 | 0.047 | 0.920 | 0.924 |
| 50-100 m | 0.323 | 0.328 | 0.024 | 0.018 | 0.858 | 0.858 |
| 100-200 m | 0.052 | 0.047 | 0.001 | 0.001 | 0.650 | 0.614 |

Recall is at BEV IoU 0.3. On aiMotive, AP3D@0.7 is limited by the train and val label difference described in [the detector results](../detectors/README.md#label-differences-between-train-and-val).

| KITTI val (3,769 frames) | n GT | APBEV@0.3 SECOND | APBEV@0.3 authors' CLOCs | APCD SECOND | APCD authors' CLOCs | AP3D@0.7 SECOND | AP3D@0.7 authors' CLOCs |
|---|---|---|---|---|---|---|---|
| All | 14,219 | 0.929 | 0.941 | 0.925 | 0.937 | 0.704 | 0.717 |
| 0-30 m | 8,714 | 0.973 | 0.972 | 0.971 | 0.971 | 0.854 | 0.852 |
| 30-50 m | 4,256 | 0.867 | 0.906 | 0.861 | 0.900 | 0.496 | 0.536 |
| 50-70.4 m | 1,249 | 0.481 | 0.622 | 0.467 | 0.596 | 0.104 | 0.156 |

**Official KITTI metric** (Car, IoU 0.7, Easy / Moderate / Hard): SECOND 3D AP 90.82 / 79.73 / 76.87, authors' CLOCs 92.75 / 81.28 / 77.70 ([`kitti_clocs_official.txt`](kitti_clocs_official.txt)). This metric cannot be computed on aiMotive.

The two datasets differ in sensors, labels and range, so compare the gain from fusion within each dataset rather than absolute scores across them.

Over KITTI's own range, 50-70.4 m on aiMotive val, CLOCs still loses (BEV AP@0.3 0.738 to 0.704), while on KITTI it gains (0.481 to 0.622). So the different result is not only about the extra range to 200 m.

### By driving condition (CLOCs mean of 3 seeds)

| | APBEV@0.3, all: SECOND | CLOCs | APBEV@0.3, 100-200 m: SECOND | CLOCs |
|---|---|---|---|---|
| **Val** | | | | |
| Overall | 0.630 | 0.599 | 0.277 | 0.267 |
| Highway | 0.639 | 0.594 | 0.318 | 0.268 |
| Urban | 0.633 | 0.603 | 0.270 | 0.274 |
| Without tunnel scenes | 0.634 | 0.608 | 0.274 | 0.274 |
| **Holdout** | | | | |
| Overall | 0.629 | 0.620 | 0.313 | 0.319 |
| Highway | 0.930 | 0.924 | 0.656 | 0.660 |
| Urban | 0.520 | 0.514 | 0.001 | 0.002 |

## Validating our implementation on KITTI

We ran our fusion code (top-2,000 candidates, cells, BEV IoU targets, network, loss, NMS, evaluator) on the authors' KITTI SECOND and 2D detections. It was trained for 3 epochs on 3,712 frames, with 3 seeds.

| KITTI val, APBEV@0.3 | SECOND | Our CLOCs | Authors' CLOCs |
|---|---|---|---|
| All | 0.929 | 0.942 (0.940-0.943) | 0.941 |
| 0-30 m | 0.973 | 0.972 (0.971-0.973) | 0.972 |
| 30-50 m | 0.867 | 0.906 (0.902-0.908) | 0.906 |
| 50-70.4 m | 0.482 | 0.621 (0.621-0.622) | 0.622 |

| KITTI val, AP3D@0.7 | SECOND | Our CLOCs | Authors' CLOCs |
|---|---|---|---|
| All | 0.704 | 0.719 (0.717-0.721) | 0.717 |
| 30-50 m | 0.496 | 0.542 (0.541-0.543) | 0.536 |
| 50-70.4 m | 0.104 | 0.160 (0.157-0.165) | 0.156 |

Full numbers are in [`kitti_our_clocs.json`](kitti_our_clocs.json). This covers the fusion code, not the aiMotive data loading or our trained detectors, which are checked in [the detector results](../detectors/README.md).

## Why CLOCs loses ground at range on aiMotive

AUC measures how well a score separates real cars from false detections, from 0.5 (no signal) to 1.0 (perfect). The boxes are those SECOND's own NMS keeps, using one trained CLOCs seed, and a real car means BEV IoU of at least 0.3 with a label.

| Val | Real cars | AUC SECOND | AUC 2D alone | AUC CLOCs | Real cars with an aligned 2D box above 0.5 |
|---|---|---|---|---|---|
| 0-30 m | 2,476 | 0.999 | 0.983 | 0.999 | 97% |
| 30-50 m | 1,408 | 0.988 | 0.965 | 0.995 | 89% |
| 50-100 m | 6,202 | 0.972 | 0.822 | 0.972 | 45% |
| 100-200 m | 4,639 | 0.909 | 0.652 | 0.873 | 8% |

"Aligned" means 2D-3D IoU of at least 0.5. The 2D column uses the best score among aligned boxes.

Camera evidence is also stronger on holdout than on val at range: 35% of real cars at 100-200 m have an aligned 2D box above 0.5 on holdout, with a 2D AUC of 0.831. Fusion learned on frames like those may therefore weight the camera too heavily for val.

### Explanations we tested and ruled out

| Explanation | Test | Result |
|---|---|---|
| Fusion is punishing LiDAR boxes with no 2D match | Count candidates with s3D of at least 0.5 and no overlapping 2D box | Almost none: 78 of about 50,000 on training frames, 3 on val. Cascade outputs up to 200 boxes per image, so nearly every 3D box overlaps one |
| Detectors look too good on training frames, so fusion over-trusts them | Train fusion on scenes the detectors never saw, cross-fitted within holdout | No improvement: holdout APBEV@0.3 0.613, against 0.620 trained on training frames and 0.629 for SECOND. Scored on val: 0.556 |
| Our implementation is at fault | Run it on KITTI, as above | It matches the authors' CLOCs |

Results are in [`diag_fusion_trust.json`](diag_fusion_trust.json) and [`diag_camera_signal.json`](diag_camera_signal.json), with their scripts alongside.

## Setup

| | |
|---|---|
| Network | 4 inputs, then 18, 36, 36 and 1 channels, 2,143 parameters |
| Training | AdamW with one-cycle LR (max 3e-3), focal loss, 3 epochs on 7,753 frames, about 2 minutes per seed |
| Targets | BEV IoU at least 0.7 positive, at most 0.5 negative, large vehicles ignored |
| Forward pass | About 1.2 ms per frame (median, fusion network only) |
| Candidates | Top 2,000 from SECOND, up to 200 from Cascade R-CNN |
| Differences from the authors' code | BEV IoU targets instead of 3D IoU, and -1 instead of -10 for the no-2D row, following our paper |

GPU training is not bit-identical between runs: retraining seed 3 reproduced its scores to within about 0.005.

Weights are in `clocs_aimotive_seed{1,2,3}.pth` and per-seed results in `aimotive_clocs_seed{1,2,3}.json`. SECOND through the same path is `aimotive_second_only.json`, which matches the detector results exactly.

## Note on an earlier invalid result

The first training run used a max-pooling function that picked the wrong cell for over 90% of candidates at full frame size, because PyTorch does not define the order of writes to repeated indices. It was caught because the training loss rose during training. It was fixed in `clocs_r/fusion.py`, with a full-size test added to `clocs_r/test_core.py`, and all results here come from the fixed code. None of the detector scores were affected.

## Reproducing

```bash
source env.sh
python -m clocs_r.cache_cells --split holdout,val,train --no_radar
python -m clocs_r.baseline_second
for s in 1 2 3; do python -m clocs_r.train_fusion --feature_set R0 --out runs/fusion/R0_s$s --seed $s --eval_splits holdout,val; done
cd second
python -m clocs_r.eval_kitti --out ../results/clocs_no_radar --only clocs
python -m clocs_r.kitti_check --out ../results/clocs_no_radar
```
