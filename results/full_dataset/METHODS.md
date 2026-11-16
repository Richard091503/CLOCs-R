# What was done: CLOCs and CLOCs-R on the full aiMotive dataset

This page covers how the full-dataset study was set up and run: the environment, the data, the code changes and the pipeline. The results and their analysis are in [README.md](README.md), and the protocol, written before any result existed, is in [PLAN.md](PLAN.md).

## Goal

Earlier results in this repository used daytime highway and urban driving only. This study answers two questions on all four aiMotive driving conditions (highway, urban, night and rain):

- How much does radar add to camera-LiDAR CLOCs?
- Is that gain larger in adverse conditions?

Training and evaluation both focus on cars beyond 50 m.

## Environment

- **Hardware:** one RTX 4090 (sm_89) on Ubuntu 22.04.
- **Install:** `bash clocs_r/install_detectors.sh` builds a conda env `clocs` with:
  - PyTorch 2.0.1 with CUDA 11.8, and CUDA 11.8 `nvcc` installed from conda. No system CUDA toolkit is needed.
  - spconv 2.3.6.
  - OpenPCDet at commit `233f849`, built in `~/OpenPCDet`.
  - mmcv-full 1.7.2 and MMDetection 2.28.2.
- **Why the 4090 needs this:** it requires CUDA 11.8 or newer, which the older stack used earlier in this repository does not support.
- **`source env.sh`** activates the env, sets `PYTHONPATH`, and sets `PYTHONNOUSERSITE=1`. Without that variable, packages in `~/.local` (pip `--user`) and ROS's Python path override the env's own versions, for example a different spconv or mmcv.
- **Build detail:** OpenPCDet's CUDA ops are built without `build_ext -j`. Parallel builds share one ninja file and fail with missing object files.

## Data

- **Download:** the aiMotive multimodal dataset (CC BY-NC-SA 4.0), unzipped to `~/datasets/aimotive`.
- **Conversion:** `clocs_r/fetch_aimotive.py --src` reads that local copy (previously it could only download) and writes the reduced format used here, with all four conditions and every annotated frame (`--train_stride 1`), to `~/datasets/aimotive_clocs`. `AIMOTIVE_ROOT` points the code at it.
- **Splits:**

| Split | Frames | Highway | Urban | Night | Rain | Used for |
|---|---|---|---|---|---|---|
| train | 18,994 | 8,924 | 6,525 | 3,117 | 428 | training the detectors and the fusion |
| holdout | 2,408 | 990 | 851 | 425 | 142 | choosing among fusion configurations |
| val | 5,181 | 2,252 | 2,040 | 444 | 445 | reported results |

Holdout is every 10th training scene within each condition (`clocs_r/splits.py`). Val is aiMotive's own validation set. It has only 2 night and 2 rain scenes.

## Focus on long range

The same weight is used everywhere, `w(r) = 1 + (2 − 1) · clip((r − 40) / 20, 0, 1)`. It is 1 below 40 m and rises linearly to 2 at 60 m and beyond (`clocs_r/range_weight.py`). It applies to every anchor or candidate, positive or negative, by its distance from the ego vehicle.

| Component | Change | Where |
|---|---|---|
| SECOND (3D, LiDAR) | Classification, box and direction losses multiplied by `w(r)` of each anchor. The training voxel cap was raised from 16,000 to 20,000, although measurement showed only 0.03% of voxels were being dropped. 40 epochs. | `clocs_r/second_head.py` (`RangeWeightedAnchorHead`), `clocs_r/configs/second_aimotive.yaml` |
| Cascade R-CNN (2D, camera) | Trained and run on images upscaled 1.5x (1920x1056) so that small far cars cover more pixels. A 2D box has no range of its own, so there is no loss weight. COCO-pretrained, 4 epochs. | `clocs_r/cascade_2d.py --scale` |
| Fusion network | The focal loss of each candidate multiplied by `w(r)`. | `clocs_r/train_fusion.py --far_weight` |

No unweighted detectors were trained, so the effect of weighting on the detectors themselves is not measured. For the fusion layer it is: one CLOCs run used the original loss (`far_weight 1`).

## Fusion models

- **CLOCs:** for each SECOND candidate, the fusion network looks at the Cascade R-CNN boxes it overlaps in the image and outputs a new score.
- **CLOCs-R:** also gets radar features for each candidate, taken from the radar returns inside a gate around the box. The gate is the box enlarged by δ = 0.5 m + 0.02 · range. Returns from 3 sweeps are accumulated.

| Name | Radar features per candidate | Residual (T2) |
|---|---|---|
| R0 | none (plain CLOCs) | no |
| R0+T2 | none | yes |
| R2 | Doppler speed (median absolute radial speed of the gated returns) | no |
| R3 | **radar points** (log of 1 + the number of gated returns) **and Doppler speed** | no |
| R2+T2 | Doppler speed | yes |
| R3+T2 | radar points and Doppler speed | yes |

With T2, the network's output is added to SECOND's own score instead of replacing it, and the last layer starts at zero. Training therefore starts from SECOND's ranking. Each configuration was trained with seeds 1–3 for 3 epochs over all training frames.

## Pipeline

All of it runs from `clocs_r/run_full.sh` once both detectors are trained. Each step writes to `runs/` (not in git) or to `results/full_dataset/`.

1. **Detectors:** SECOND (`clocs_r/train_second.py`, about 5 h) and Cascade R-CNN (`clocs_r/cascade_2d.py train`, about 3 h), trained at the same time on the one GPU.
2. **Candidates:** SECOND's top 2,000 boxes before NMS and Cascade R-CNN's top 200 boxes, exported for every frame of train, holdout and val (30 min).
3. **Detectors alone:** scored on holdout and val (`clocs_r/eval_detectors.py` → `detectors/`).
4. **Fusion inputs:** candidate-pair tensors and radar features cached (`clocs_r/cache_cells.py`). Pair coordinates are stored as float16 so that all 26,583 frames fit in memory.
5. **Selection:** every configuration trained with 3 seeds and scored on holdout (`clocs_r/tune_fusion.py` → `selection.json`, 80 min). `clocs_r/select_full.py` then picked the CLOCs-R model by the pre-registered rule: highest center-distance AP beyond 50 m on holdout. R2+T2 won (`selected.json`). Val was not read during this step.
6. **Val:** the saved networks scored on val once (`final.json`, `final_far_weight_1.json`).
7. **Report:** per-scene scores (`per_scene.py`), then tables and figure (`make_report.py` → `tables.md`, `figures/`).

## The R3+T2 rerun (radar points and Doppler)

After the first round, the CLOCs-R model was switched to the one that uses both radar features: R3+T2, the number of radar points plus Doppler speed. It was the second-best candidate on holdout (0.4335, against 0.4353 for R2+T2).

- **Retraining:** R3+T2 and its radar-free twin R0+T2 were trained again with 3 new seeds (4, 5, 6) and scored on holdout and val (`tune_fusion.py --seeds 4,5,6` → `rerun_r3.json`).
- **Fair comparison:** R2+T2 (Doppler only) got the same 3 extra seeds (`rerun_r2.json`), so the two radar feature sets can be compared against the same twin on equal terms.
- **Pooling:** all three were pooled with seeds 1–3 from the first round, giving 6 seeds per model and 6 seed-paired radar-gain estimates.
- **Reports:** `per_scene.py --models R3+T2,R2+T2,R0+T2 --seeds 1,2,3,4,5,6 --suffix _r3` and `make_report.py --model R3+T2 --twin R0+T2 --extra rerun_r3.json,rerun_r2.json --suffix _r3` write `per_scene_val_r3.json`, `tables_r3.md` and `figures/far_by_condition_r3.png`.
- **Pre-registration:** this choice of model is not the pre-registered one. R2+T2 remains the model selected under [PLAN.md](PLAN.md), and its tables are kept in `tables.md`.

## The full ablation (second round), which sets the final configurations

The repository's whole ablation process was then repeated on the full dataset. The rules were fixed in [ablation/PLAN.md](ablation/PLAN.md) before the first run, and the work is driven by `clocs_r/ablation_full.py`.

1. **Gate:** R3 at each of 9 radar gates (δ0 ∈ {0.25, 0.5, 1.0} m × κ ∈ {0.01, 0.02, 0.04}); the highest holdout mean wins.
2. **Feature sets:** R0–R7 at that gate; the best of R1–R7 on holdout becomes the base set S.
3. **Fusion changes:** T1–T5 each tested alone on S.
   - A change is adopted if it beats S by more than the standard deviation of S's 3 seeds.
   - If two or more are adopted, their combination is also tested.
   - The best of these is the **final CLOCs-R**. Its radar-free twin, with the radar-only changes removed, is the **final CLOCs**.
4. **Val:** every network is scored on val once. The two finals get 3 more seeds (6 in total), and are also scored with the camera removed. Latency is measured for R0–R7.

Everything is resumable. Outputs go to `results/full_dataset/ablation/` (`decisions.json` records each choice) and weights to `runs/ablation/`. First-round runs with identical settings are reused rather than retrained. This includes seeds 4–6 of the two finals, which had been trained for the R3 rerun.

**Outcome:** gate δ0 = 0.5 m, κ = 0.02, feature set R2 and change T2, so the **final CLOCs-R is R2+T2 and the final CLOCs is R0+T2**. This is the same pair the first round selected. Results and the per-step findings are in [ablation/README.md](ablation/README.md).

The final report on 6 seeds:
- `per_scene.py --models R2+T2,R0+T2 --seeds 1,2,3,4,5,6 --weights runs/ablation/sets_d0.5_k0.02 --suffix _final`
- `make_report.py --model R2+T2 --twin R0+T2 --extra rerun_r2.json,rerun_r3.json --selected '' --suffix _final`

These write `per_scene_val_final.json`, `tables_final.md` and `figures/far_by_condition_final.png`.

## Loss curves

`plot_losses.py` draws the training loss of all three trained parts from logs kept in git:
- SECOND, from `logs/second_train.txt`.
- Cascade R-CNN, from `logs/cascade_train.json`.
- The fusion network, from the loss logs in the tune_fusion result files: every run, every seed, logged every 200 steps.

It writes `figures/loss_second.png`, `figures/loss_cascade.png` and `figures/loss_fusion.png`. By default the fusion plot shows the final pair; `--fusion A,B` picks other configurations. All are training losses; no validation loss was logged during training.

## Example frames

`examples.py` picks val frames where the models succeed and fail, and renders each one (`figures/examples/`, described in [EXAMPLES.md](EXAMPLES.md)):
- **Thresholds:** each model runs at the score threshold that maximizes F1 on holdout. A detection is correct within 2 m of a car's center.
- **Strict misses:** a model only counts as missing a car (or dropping a false alarm) when it has no detection above threshold within 4 m. A slightly misplaced box therefore does not count as a miss.
- **Radar categories:** these require at least one radar return in the gate, so the difference can come from radar evidence.
- **Variety:** frames come from different scenes, and from as many driving conditions as possible.

## Metrics

- **Center-distance AP (main metric):** a detection matches a label when their BEV centers are within a distance threshold. The score is the mean of the APs at 0.5, 1, 2 and 4 m (the nuScenes thresholds), so it rewards finding a car rather than tight box overlap. Also reported: BEV AP at IoU 0.3, AP3D at IoU 0.5 and 0.7, and recall.
- **Distance bins:** 0–30, 30–50, 50–100 and 100–200 m, and **beyond 50 m** (50–200 m, the study's focus).
- **Condition groups:** all, clear (highway + urban), adverse (night + rain), each condition alone, and all scenes except tunnels.
- **Radar gain:** CLOCs-R minus its radar-free twin, computed seed by seed (seed i against seed i), reported as the mean with the range across seed pairs.

## Code changes (relative to the previous commit)

| File | Change |
|---|---|
| `env.sh` | Repo-relative paths, conda env, isolation from user-site and ROS packages |
| `clocs_r/install_detectors.sh` | Rewritten for CUDA 11.8, PyTorch 2.0.1, OpenPCDet and MMDetection 2.28 on current GPUs |
| `clocs_r/fetch_aimotive.py` | Reads a local copy of the dataset (`--src`), all four conditions (`--odds`) |
| `clocs_r/aimotive.py`, `clocs_r/splits.py` | Dataset root from `AIMOTIVE_ROOT`; all four conditions by default (`CLOCS_ODDS` restores the old set) |
| `clocs_r/range_weight.py` (new) | The range weight `w(r)` |
| `clocs_r/second_head.py` (new) | `RangeWeightedAnchorHead`, SECOND's anchor head with range-weighted losses |
| `clocs_r/configs/second_aimotive.yaml` | Uses the weighted head; voxel cap 20,000 |
| `clocs_r/train_second.py` | Registers the head; `--resume`; saves every epoch with optimizer state |
| `clocs_r/cascade_2d.py` | `--scale` (default 1.5), `--max_iters` and `--limit` for quick tests |
| `clocs_r/train_fusion.py` | Range-weighted fusion loss (`--far_weight`); scores per condition with a beyond-50 m bin |
| `clocs_r/cache_cells.py` | float16 pair coordinates (memory) |
| `clocs_r/eval_detectors.py` | Same condition groups and bins as the fusion |
| `clocs_r/tune_fusion.py` | All feature sets R0–R7; `--far_weight`; `--from_weights` to score saved networks without retraining; `--seeds`; `--gate`; `--nocam` |
| `clocs_r/select_full.py` (new) | The pre-registered selection rule |
| `clocs_r/run_full.sh` (new) | The whole pipeline after detector training |
| `clocs_r/ablation_full.py` (new) | The second-round ablation: gate, feature sets, fusion changes, val, camera removal, latency |
| `clocs_r/radar_latency.py` | Reads tune_fusion's weight layout |
| `results/full_dataset/` (new) | Plan, methods, dataset analysis, results, examples, loss curves, raw JSON results, logs |

## Logs and weights

- **Logs:** training and pipeline logs are in [`logs/`](logs/): `second_train.txt`, `cascade_train.json` (MMDetection log), `run_full.txt`, `rerun_r3.txt`, `rerun_r2.txt` and `ablation_full.txt`.
- **Weights:** trained weights are too large for git and stay in `runs/`:
  - SECOND: `runs/second/checkpoint_epoch_40.pth`
  - Cascade R-CNN: `runs/cascade/epoch_4.pth`
  - Fusion networks: `runs/fusion_full/{config}_s{seed}.pth` (first round and reruns), `runs/ablation/gate_*/` and `runs/ablation/sets_d0.5_k0.02/` (second-round ablation)
- **License:** the weights are derived from aiMotive data and fall under its CC BY-NC-SA 4.0 license (non-commercial).
