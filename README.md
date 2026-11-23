# CLOCs-R: one CLOCs network over LiDAR, camera and radar detectors (LCR)

**Question:** does radar help a LiDAR detector find cars beyond 50 m?

**LCR** answers it within CLOCs. CLOCs (Pang et al., IROS 2020) rescores a LiDAR detector's 3D candidates using
another detector's boxes. LCR gives one CLOCs network the outputs of three independent detectors at once:

```
 LiDAR  ──► SECOND ─────────── 3D candidates (boxes, scores) ────────┐
 camera ──► Cascade R-CNN ──── 2D boxes ──► camera cells   [IoU, s2D, s3D, d, −, −] ──┤
 radar  ──► D2 (PointPillars) ─ BEV boxes ─► radar cells   [−, −, s3D, d, affinity, s_radar] ──┤
                                                                     ▼
                     one CLOCs network: 1×1 convolutions over every cell, max per candidate,
                     added to SECOND's logit  ──►  new score for each SECOND candidate  ──►  NMS
```

- **Camera cells.** Each SECOND candidate is paired with every Cascade R-CNN box that overlaps its projection, keeping
  2D boxes at least as strong as the frame's 40th best (K40).
- **Radar cells.** It is also paired with every D2 box within 2 m: affinity 1 − d/2 m, plus the radar score.
- **Absent sensor.** The places of the absent sensor hold −1. A candidate with no pair keeps a reserved cell.
- **Scope.** The network only rescores SECOND's candidates and never adds boxes.
- **Size.** 2,179 parameters.

Data: the aiMotive multimodal dataset, all conditions (highway, urban, night, rain). Every detector and every fusion
network is trained on `fit`; every setting was chosen on `select`; val was scored once per model. The structure of every
model, the splits and the metric are in [docs/MODELS.md](docs/MODELS.md).

## Results

APCD (centre-distance AP for cars, averaged over 0.5–4 m), mean of 6 seeds. Full tables, conditions, seed-paired
differences and bootstrap intervals: [results/lcr/REPORT.md](results/lcr/REPORT.md).

| Model | `select` 0–30 | 30–50 | 50–200 | 100–200 | val 0–30 | 30–50 | 50–200 | 100–200 |
|---|---|---|---|---|---|---|---|---|
| SECOND alone (L) | 0.9072 | 0.7797 | 0.5639 | 0.4052 | 0.8971 | 0.6206 | 0.4455 | 0.2841 |
| Camera CLOCs (LC) | 0.9066 | 0.7864 | 0.5807 | 0.4096 | 0.9158 | 0.6304 | 0.4264 | 0.2789 |
| Radar CLOCs (LRd) | 0.9080 | 0.7759 | 0.5753 | 0.4162 | 0.8913 | 0.6213 | 0.4415 | 0.2858 |
| **LCR (one network, all three)** | **0.9095** | **0.7812** | **0.5817** | **0.4125** | **0.9132** | **0.6294** | **0.4271** | **0.2779** |
| LCR-D (far camera dropout in training) | 0.9107 | 0.7790 | 0.5797 | 0.4182 | 0.9087 | 0.6274 | 0.4347 | 0.2801 |

**What the results show** (scene-bootstrap 95% intervals in brackets):
- **On `select`, LCR is above SECOND alone on average at every range,** most beyond 50 m: +0.018 at 50–200 m, with
  all 6 seeds higher [−0.004, +0.043]. At 30–50 m, only 2 of 6 seeds are higher.
- **On val, LCR beats SECOND alone within 50 m:** +0.016 at 0–30 m and +0.009 at 30–50 m (6/6 seeds each).
- **On val, LCR falls below SECOND alone beyond 50 m:** −0.018 [−0.033, −0.005] at 50–200 m.
- **LCR's radar cells add almost nothing over camera CLOCs:** LCR − LC is +0.001 [−0.002, +0.005] on `select` and
  +0.001 [−0.004, +0.005] on val at 50–200 m. The single network learns to lean on the camera.
- **Radar does carry information beyond 50 m when it has its own network.**
  - On `select`, LRd − SECOND is +0.0114 [+0.0032, +0.0185] at 50–200 m. On val it is −0.0040 [−0.0146, +0.0050].
  - A blend of the camera and radar networks (B, 72% / 28%) beats LCR on both splits. LCR − B at 50–200 m is
    −0.012 [−0.017, −0.004] on `select` and −0.016 [−0.022, −0.009] on val.
  - LCR is the main model here because it is one network, the simplest form of CLOCs over three detectors. The
    pre-registered rule had chosen B (see REPORT.md, "Pre-registered rules").
- **Hiding the camera beyond 50 m in training (LCR-D) moves the network towards radar,** +0.008 on val at 50–200 m
  (6/6 seeds). Its intervals include 0 on both splits, and it loses 0.002 at 30–50 m on `select`, so its rule kept LCR.
- **Likely cause, not yet tested:** Cascade R-CNN and D2 are trained on the same `fit` scenes as the fusion network.
  A camera detector looks far more reliable on the scenes it was trained on. The project's earlier Cascade R-CNN had
  a 2D recall@0.5 of 92% for 100–200 m cars on its own training scenes, against 23% on val; the `fit`-trained one used
  here has 35% on `select` and 23% on val. Cross-fitted detector outputs would test this.

## Reproduce

```bash
bash clocs_r/install_detectors.sh     # conda env `clocs`: PyTorch 2.0.1 + CUDA 11.8, OpenPCDet (pinned), MMDetection 2.28
source env.sh                         # every later command assumes this; AIMOTIVE_ROOT defaults to ~/datasets/aimotive_clocs
python -m clocs_r.fetch_aimotive --src <extracted aiMotive> --out ~/datasets/aimotive_clocs --train_stride 1
python -m clocs_r.test_core           # quick checks (no GPU needed)
nohup setsid bash clocs_r/run_lcr.sh >> runs/lcr/run.log 2>&1 &   # detectors, candidates, cells, D2, fusion, val, bootstrap
python results/lcr/report.py          # results/lcr/REPORT.md from the stored results
```

`run_lcr.sh` writes to `runs/lcr/` and `results/lcr/rerun/`, and skips every step whose output exists. Most of the time
goes to training the three detectors. The trained fusion networks are in `results/lcr/weights/`. With the cached cells
in `runs/lcr/`, they can be re-scored without training, for example LCR on val:

```bash
# only if runs/lcr/cells/{fit,select,val}_base.pkl are missing (a cache; rebuilt byte-identical in about 5 min):
python -m clocs_r.cache_cells --split select,val,fit --no_radar --out runs/lcr/cells --cand3d runs/lcr/cand/3d --cand2d runs/lcr/cand/2d
cp results/lcr/weights/*.pth runs/lcr/fusion/
python -m clocs_r.tune_fusion --configs R0+T2+K40+RC --splits val --seeds 1,2,3 --from_weights --out results/lcr/rerun/lcr_val.json
```

## Repository

| Path | Contents |
|---|---|
| `clocs_r/` | the LCR pipeline: detectors (`train_second.py`, `cascade_2d.py`, `radar_dataset.py`, `radar_clutter.py`), candidates and cells (`export_3d.py`, `cache_cells.py`, `visibility.py`, `radar_pairs.py`), fusion (`fusion.py`, `train_fusion.py`, `tune_fusion.py`), evaluation (`metrics.py`, `eval_detectors.py`, `radar_candidates.py`, `bootstrap.py`), `run_lcr.sh`, `test_core.py` |
| `results/lcr/` | `REPORT.md`, `report.py`, the raw results in `data/` and the 24 trained fusion networks in `weights/` |
| `docs/MODELS.md` | every model's structure: the five here, and those on the `thesis` branch |

The **`thesis`** branch keeps the whole project record unchanged:
- the earlier CLOCs-R with radar point features;
- the range switch, the visibility and pairing variants, gated fusion, oracle injection and radar-sourced candidates;
- the radar detectors D1 and D3 (RadarDistill), the selector, the blend and the oracle ceiling;
- cross-fitting, and the KITTI reproduction of the original CLOCs;
- every pre-registered rule, decision, log and result.

## Credits

- **CLOCs:** Pang, Morris and Radha, *CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection*, IROS 2020
  ([code](https://github.com/pangsu0613/CLOCs)). This repository started as a fork of it (MIT license, `LICENSE`).
- **Toolboxes:** [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) for SECOND and PointPillars, and
  [MMDetection](https://github.com/open-mmlab/mmdetection) for Cascade R-CNN.
- **Data:** the [aiMotive multimodal dataset](https://github.com/aimotive/aimotive_dataset).
