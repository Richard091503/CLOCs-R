# Next round

The pre-registered rule for each phase is in [PLAN.md](PLAN.md). Results are in `phase*/`.
Every command runs from the repository root after `source env.sh`.

## Phase 0: environment, data and reproduction (2026-11-18)

**Result: passes.** Both final models trained on detectors retrained here are within 2 standard errors of the RTX 4090
reference on val beyond 50 m (6 seeds each):

| Model | 4090 (mean ± SD) | This run (mean ± SD) | Difference | 2 SE limit | Reproduces |
|---|---|---|---|---|---|
| SECOND alone | 0.453 | 0.457 | +0.004 | (no threshold) | |
| R0+T2 (CLOCs) | 0.4144 ± 0.0080 | 0.4208 ± 0.0070 | +0.0064 | 0.0087 | yes |
| R2+T2 (CLOCs-R) | 0.4224 ± 0.0081 | 0.4199 ± 0.0097 | −0.0026 | 0.0103 | yes |

Per seed (val, center-distance AP beyond 50 m, seeds 1–6):

| | Seed 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| R0+T2, 4090 | 0.4188 | 0.4238 | 0.4051 | 0.4098 | 0.4220 | 0.4073 |
| R0+T2, here | 0.4216 | 0.4272 | 0.4303 | 0.4150 | 0.4189 | 0.4120 |
| R2+T2, 4090 | 0.4224 | 0.4278 | 0.4296 | 0.4068 | 0.4235 | 0.4244 |
| R2+T2, here | 0.4208 | 0.4268 | 0.4119 | 0.4057 | 0.4217 | 0.4322 |
| Radar gain, 4090 | +0.0036 | +0.0040 | +0.0245 | −0.0030 | +0.0015 | +0.0172 |
| Radar gain, here | −0.0008 | −0.0004 | −0.0184 | −0.0093 | +0.0028 | +0.0202 |

The radar gain on val beyond 50 m is −0.001 here: 2 of 6 seed pairs are positive, against +0.008 and 5 of 6 on the 4090.
On holdout it is +0.006 here, 6 of 6 positive, against +0.008 and 6 of 6.
Both fused models are still below SECOND alone beyond 50 m on val, as in the 4090 run.

The committed 4090 fusion networks, scored on this run's candidates, give 0.4129 (R0+T2) and 0.4222 (R2+T2).
That is within 0.002 of their reported values, so the new detectors' candidates are close to the old ones.

The full tables are in [phase0/report.md](phase0/report.md): every condition and distance bin on val and holdout, the
seed-against-seed radar gain, and APBEV@0.3, AP3D@0.5 and recall. The numbers behind the rule are in
[phase0/summary.json](phase0/summary.json).

**Deviations from the 4090 run.** All were decided before any result; see PLAN.md.

- The dataset was converted on another machine and copied here, because Kaggle returns 404 to this host.
- Both detectors were retrained here.
- Cascade R-CNN used batch 2 with 2-step gradient accumulation (`--accum 2`).
- SECOND used 6 data-loader workers instead of 12.

**Incidents.**

- The first run of fusion seeds 4–6 crashed partway through, with `CUDA error: unspecified launch failure` in
  `loss.backward()`. The kernel log showed no GPU fault.
  - The partial outputs were set aside in `runs/next_round/failed_s456/`, and the same command was rerun unchanged.
  - The rerun finished.
- 19 val frames have no LiDAR points. Exporting at batch 4 handles them. A batch of 1 or 2 could be all-empty and trip
  spconv's assertion.

**Code changes.** The existing configurations are unchanged.

- `clocs_r/aimotive.py` reads `meta.pkl` files pickled under NumPy 2.
- `clocs_r/cascade_2d.py` has a new `--accum` flag (default 1).
- `env.sh` and `install_detectors.sh` detect the GPU architecture. They also install GCC 11, because nvcc 11.8 rejects GCC 13.
- `run_full.sh` takes `CLOCS_OUT`, so a rerun writes to its own folder.

**Reproduce.**

```bash
bash clocs_r/install_detectors.sh
source env.sh                                  # AIMOTIVE_ROOT defaults to ~/datasets/aimotive_clocs
# dataset, on a machine that can reach Kaggle, then copied to $AIMOTIVE_ROOT
python -m clocs_r.fetch_aimotive --src ~/datasets/aimotive --out ~/datasets/aimotive_clocs --train_stride 1
# detectors (Cascade R-CNN took about 2.5 h on the 10 GB A100 slice)
python -W ignore -m clocs_r.train_second --out runs/second --workers 6 > runs/second/console.log 2>&1
python -W ignore -m clocs_r.cascade_2d train --out runs/cascade --epochs 4 --accum 2 > runs/cascade/console.log 2>&1
# candidates and detectors alone
python -W ignore -m clocs_r.export_3d --ckpt runs/second/checkpoint_epoch_40.pth --out runs/candidates3d --splits val,holdout,train
python -W ignore -m clocs_r.cascade_2d export --out runs/cascade --ckpt runs/cascade/epoch_4.pth --candidates runs/candidates2d --splits val,holdout,train
python -W ignore -m clocs_r.eval_detectors --splits holdout,val --out results/next_round/phase0/detectors
# fusion cells and radar features (7.2 GB), then the final pair
python -W ignore -m clocs_r.cache_cells --split holdout,val,train
F="--far_weight 2 --gate 0.5,0.02 --splits holdout,val"
python -W ignore -m clocs_r.tune_fusion --configs R2+T2,R0+T2 $F --seeds 1,2,3 --weights runs/next_round/phase0_fusion --out results/next_round/phase0/fusion_s123.json
python -W ignore -m clocs_r.tune_fusion --configs R2+T2,R0+T2 $F --seeds 4,5,6 --weights runs/next_round/phase0_fusion --out results/next_round/phase0/fusion_s456.json
# side check: the committed 4090 fusion networks on the new candidates
mkdir -p runs/next_round/committed_4090 && cp results/full_dataset/weights/*.pth runs/next_round/committed_4090/
python -W ignore -m clocs_r.tune_fusion --configs R2+T2,R0+T2 $F --seeds 1,2,3,4,5,6 --from_weights --weights runs/next_round/committed_4090 --out results/next_round/phase0/committed_4090_weights.json
# report and rule
python results/next_round/phase0_report.py > results/next_round/phase0/report.md
```

## Phase 1: a selection split (2026-11-18)

**Split.** The split is `select`, defined in `clocs_r/splits.py` (`SELECT_RECORDINGS`). The rest of aiMotive's training
scenes are `fit`; `train`, `holdout` and `val` are unchanged. The split is 11 whole training recordings chosen by the
PLAN.md rule: per condition, about 15% of frames and 20% of cars at 100–200 m. No drive is on both sides. In the old
holdout, 14 of 16 recordings also have scenes in `train`.

| Condition | Split | Scenes | Recordings | Frames | 0-30 m | 30-50 m | 50-100 m | 100-200 m |
|---|---|---|---|---|---|---|---|---|
| all | train | 134 | 43 | 18,994 | 22,178 | 24,439 | 37,146 | 11,779 |
| all | holdout | 17 | 16 | 2,408 | 3,691 | 4,039 | 5,847 | 938 |
| all | **select** | 24 | 11 | 3,409 | 3,996 | 5,275 | 6,874 | 2,760 |
| all | **fit** | 127 | 34 | 17,993 | 21,873 | 23,203 | 36,119 | 9,957 |
| all | val | 25 | 8 | 5,181 | 3,111 | 1,895 | 8,148 | 8,224 |
| highway | select / holdout / val | 10 / 7 / 12 | 3 / 6 / 4 | 1,419 / 990 / 2,252 | 969 / 405 / 361 | 1,519 / 632 / 310 | 1,725 / 1,482 / 653 | 778 / 461 / 1,680 |
| urban | select / holdout / val | 8 / 6 / 9 | 4 / 6 / 1 | 1,136 / 851 / 2,040 | 2,594 / 2,622 / 2,113 | 2,750 / 2,744 / 1,160 | 3,454 / 3,651 / 6,255 | 1,271 / 464 / 5,097 |
| night | select / holdout / val | 4 / 3 / 2 | 3 / 3 / 2 | 569 / 425 / 444 | 388 / 664 / 382 | 813 / 521 / 212 | 1,289 / 714 / 630 | 488 / 13 / 719 |
| rain | select / holdout / val | 2 / 1 / 2 | 1 / 1 / 1 | 285 / 142 / 445 | 45 / 0 / 255 | 193 / 142 / 213 | 406 / 0 / 610 | 223 / 0 / 728 |

The per-condition rows for every split, including `train` and `fit`, are in [phase1/counts.md](phase1/counts.md).

**What the split costs.**

- `fit` keeps 95% of training frames, 85% of the training cars at 100–200 m (9,957 against 11,779), and 285 rain frames.
- `fit` has no rain cars beyond 100 m, because rain has only two training recordings and only one of them has far cars.
- `select` has 2,760 cars at 100–200 m, about 3 times the old holdout's 938. Val has 8,224.

**The current detectors cannot be scored fairly on `select`.** They were trained on 22 of its 24 scenes. Detectors
alone, from the Phase 0 detectors (`phase1/detectors/`):

| | train (seen) | select (22 of 24 scenes seen) | holdout (unseen) | val (unseen) |
|---|---|---|---|---|
| SECOND APCD beyond 50 m | 0.728 | 0.804 | 0.434 | 0.457 |
| SECOND APCD beyond 50 m, rain | 0.993 | 0.991 | (no cars) | 0.513 |
| Cascade R-CNN AP2D@0.5 | 0.864 | 0.869 | 0.498 | 0.396 |
| Cascade R-CNN candidate recall@0.5, 100–200 m | 0.918 | 0.860 | 0.283 | 0.226 |

So `select` needs detectors trained on `fit` only. The same gap also affects the current fusion training. Fusion trains
on `train`, where Cascade R-CNN finds 92% of cars at 100–200 m. On val it finds 23%.

**Reproduce.**

```bash
python -m clocs_r.test_core                                   # includes the select/fit partition checks
python results/next_round/phase1_counts.py > results/next_round/phase1/counts.md
python -W ignore -m clocs_r.eval_detectors --splits select,train --out results/next_round/phase1/detectors
```

### Phase 1, continued: detectors trained on `fit`, cross-fitting, and val once (2026-11-19)

The run follows the PLAN.md addendum. All tables are in [phase1/report.md](phase1/report.md), and the rule's inputs
are in [phase1/decision.json](phase1/decision.json).

**Detectors trained on `fit`, alone.** `select` is now unseen.

| | select | val |
|---|---|---|
| SECOND, APCD beyond 50 m | 0.564 | 0.446 (0.457 with the Phase 0 detectors) |
| Cascade R-CNN, candidate recall@0.5 at 100–200 m | 0.353 | 0.227 |

**Rule (select, APCD beyond 50 m, seeds 1–3): the in-sample arm is kept.**

| Model | In-sample | Cross-fitted | Difference | SD of cross-fitted |
|---|---|---|---|---|
| R2+T2 | 0.5740 | 0.5731 | −0.0009 | 0.0019 |
| R0+T2 | 0.5736 | 0.5641 | −0.0096 | 0.0050 |

Cross-fitting did not help R2+T2, but it changed what the network does with radar. On `select`, the radar gain beyond
50 m is +0.009 with cross-fitted inputs (3 of 3 seeds positive), against +0.000 in-sample (1 of 3). It is +0.028 on
highway and +0.051 in rain at 100–200 m, both 3 of 3. Without radar, cross-fitting costs 0.010. One reading is that once
the camera evidence in training is as weak as it is on unseen scenes, the network leans on radar. This is a `select`-only
observation: under the rule, cross-fitting was not scored on val.

**Val, once, in-sample arm, 6 seeds (APCD beyond 50 m).** These are the reference for later phases.

| Model | Mean ± SD | Radar gain, seed against seed |
|---|---|---|
| SECOND alone | 0.446 | |
| R0+T2 | 0.3984 ± 0.0108 | |
| R2+T2 | 0.3986 ± 0.0034 | +0.000 (−0.009 to +0.010), 3 of 6 positive |

- Fusion is 0.047 below SECOND beyond 50 m on val. With the Phase 0 detectors the gap was 0.037.
- Within 50 m fusion still helps: +0.019 at 0–30 m and +0.008 to +0.014 at 30–50 m.
- In rain, the radar gain beyond 50 m is −0.065 (1 of 6). `fit` has no rain cars beyond 100 m, the trade-off named
  in Phase 1.

**`select` and val disagree beyond 50 m.**

- On `select`, fusion beats SECOND beyond 50 m: 0.574 against 0.564. On val it loses by 0.047.
- On both, fusion loses at 100–200 m (`select`: 0.385 against 0.405) and gains at 50–100 m.
- `select`'s 50–200 m bin is 71% cars at 50–100 m, while val's is 50%. So a rule on the pooled bin favours changes that
  help at 50–100 m.
- Later phases' rules should look at 100–200 m on its own as well.

**Reproduce.**

```bash
bash results/next_round/phase1_run.sh            # about 21 h: 4 detector pairs, candidates, cells, both arms, the rule, val
python results/next_round/phase1_report.py > results/next_round/phase1/report.md
```

**Disk.** The rejected arm's cells and candidates were deleted afterwards. `phase1_run.sh` rebuilds them from the fold
detectors in `runs/next_round/phase1/det/`. Free disk is 10 GB.

## Phase 2: the range switch (2026-11-19)

No training. The switch rescores each candidate with a Phase 1 network (in-sample arm, seeds 1–6) under 50 m of BEV
range and keeps SECOND's own score from 50 m on, before NMS. Code: `clocs_r/range_switch.py`. Full tables:
`phase2/report.md`.

**APCD, mean of 6 seeds.**

| Split | Distance | SECOND | R0+T2 | R2+T2 | switch(R0+T2) | switch(R2+T2) |
|---|---|---|---|---|---|---|
| select | all | 0.719 | 0.720 | 0.724 | 0.720 | 0.722 |
| select | 0–30 | 0.907 | 0.907 | 0.907 | 0.906 | 0.906 |
| select | 30–50 | 0.780 | 0.785 | 0.789 | 0.779 | 0.785 |
| select | 50–100 | 0.628 | 0.649 | 0.651 | 0.626 | 0.625 |
| select | 100–200 | 0.405 | 0.386 | 0.377 | 0.405 | 0.405 |
| select | **50–200** | **0.564** | 0.574 | 0.574 | 0.563 | 0.562 |
| val | all | 0.540 | 0.502 | 0.509 | 0.544 | **0.548** |
| val | 0–30 | 0.897 | 0.916 | 0.914 | 0.916 | 0.914 |
| val | 30–50 | 0.621 | 0.629 | 0.635 | 0.626 | 0.633 |
| val | 50–100 | 0.599 | 0.554 | 0.553 | 0.598 | 0.598 |
| val | 100–200 | 0.284 | 0.248 | 0.243 | 0.284 | 0.284 |
| val | **50–200** | **0.446** | 0.398 | 0.399 | 0.445 | 0.445 |

- **On val, switch(R2+T2) is the best model overall**: 0.548 against SECOND's 0.540 and R2+T2's 0.509. It keeps
  fusion's gain within 50 m (+0.017 at 0–30 m, +0.012 at 30–50 m) and gives up nothing beyond.
- **Beyond 50 m it is SECOND to within 0.002, not exactly.** 100–200 m matches SECOND to three decimals on both
  splits. The whole loss is at 50–100 m (select −0.002, val −0.001). Fused scores just inside 50 m change which boxes
  just past 50 m survive NMS.
- **NMS differences beyond 50 m are 33–49k per seed on select and 55–83k on val.** Most of these come from the NMS caps
  (top 1,000 before, 100 after), not from boundary overlap. Fusion changes the near scores, so a different set of
  low-score far boxes makes the cut. Far scores are unchanged, so this only adds or drops boxes at the bottom of the
  ranking: recall@BEV0.3 beyond 50 m rises from 0.776 to 0.778 (select) and from 0.739 to 0.743 (val).
- **Radar gain inside the switch.** Overall it is +0.002 on select and +0.005 on val (6 of 6 both). Most of it comes from
  30–50 m (+0.006 select, +0.007 val, 6 of 6). In rain it is −0.045 at 0–30 m on val (0 of 6), the same rain weakness as
  Phase 1.

**The bar for later phases.** On `select`, under the standing rule:

| Bin | switch(R2+T2) | switch(R0+T2) |
|---|---|---|
| 50–200 m | 0.562 | 0.563 |
| 100–200 m | 0.405 | 0.405 |

A new model must beat both bins by more than its seed SD. Phase 1's R2+T2 clears the pooled bin (0.574) and fails
100–200 m (0.377).

**Reproduce** (about 2.5 h on the A100 slice):

```bash
python -m clocs_r.range_switch --weights runs/next_round/phase1/fusion_insample --cells runs/next_round/phase1/cells_in \
    --cand3d runs/next_round/phase1/cand_in/3d --cand2d runs/next_round/phase1/cand_in/2d --out results/next_round/phase2/switch.json
python results/next_round/phase2_report.py > results/next_round/phase2/report.md
```

## Phase 3: telling the network when the camera can see (2026-11-19)

Nine changes, each behind a configuration token, were ablated on R2+T2 on `select` by the rule in PLAN.md. Code:
`clocs_r/visibility.py` (H, P), `fusion.filter_pairs` (I1, I2, K40), and `train_fusion.fit` (`cam_dropout` for D2/D3,
`frame_weights` for S50/S100). Full tables: `phase3/report.md`.

R2+T2 retrained with the new code reproduces Phase 1 exactly on `select`, seed for seed: 0.5760, 0.5762, 0.5699.

**Stage A (select, seeds 1–3, APCD, change from R2+T2).**

| Token | 50–200 m | 100–200 m | Passes |
|---|---|---|---|
| H, projected 2D height | +0.0018 (SD 0.0011) | +0.0018 (SD 0.0027) | no |
| P, LiDAR points | +0.0023 (SD 0.0041) | −0.0045 | no |
| I1, pair IoU > 0.1 | −0.0010 | −0.0040 | no |
| I2, pair IoU > 0.25 | +0.0003 | −0.0049 | no |
| **K40, top 40 2D boxes** | **+0.0043 (SD 0.0019)** | **+0.0186 (SD 0.0051)** | **yes** |
| D2, camera dropout 0.2 | −0.0185 | −0.0233 | no |
| D3, camera dropout 0.3 | −0.0139 | −0.0070 | no |
| S50, oversample frames with a car beyond 50 m | −0.0003 | −0.0014 | no |
| S100, oversample frames with a car beyond 100 m | +0.0006 | −0.0027 | no |

- Only K40 passed, so there was no Stage B, and the final configuration is **R2+T2+K40**.
- It has the same 2,161 parameters. Only the input changes: about half as many cells per frame. The low-score 2D
  boxes no longer pair with far candidates, so more of those candidates use the no-camera row.
- Camera dropout hurt in every bin.

**Final, 6 seeds, APCD beyond 50 m (mean ± SD).**

| Model | select 50–200 | select 100–200 | val 50–200 | val 100–200 |
|---|---|---|---|---|
| SECOND alone | 0.564 | 0.405 | **0.446** | **0.284** |
| R0+T2 (Phase 1) | 0.574 | 0.386 | 0.398 | 0.248 |
| R2+T2 (Phase 1) | 0.574 | 0.377 | 0.399 | 0.243 |
| switch(R2+T2) (Phase 2) | 0.562 | 0.405 | 0.445 | 0.284 |
| R0+T2+K40 | **0.5807 ± 0.0013** | **0.4096 ± 0.0033** | 0.4264 ± 0.0021 | 0.2789 ± 0.0038 |
| **R2+T2+K40 (final)** | 0.5782 ± 0.0015 | 0.3958 ± 0.0038 | 0.4231 ± 0.0027 | 0.2722 ± 0.0032 |

- **K40 closes about half of fusion's gap to SECOND on val beyond 50 m**: R2+T2 goes from 0.399 to 0.423 (SECOND
  0.446). At 100–200 m it goes from 0.243 to 0.272 (SECOND 0.284). Within 50 m it keeps fusion's gain: +0.016 at
  0–30 m and +0.015 at 30–50 m over SECOND.
- **It still loses to SECOND and to the switch on val beyond 50 m**, by 0.022 (50–200 m) and 0.012 (100–200 m). Over
  all distances on val the switch remains best: 0.548 against 0.527.
- **Does anything beat the range switch on `select`?**
  - The final R2+T2+K40 does not. It is ahead at 50–200 m (+0.016) and behind at 100–200 m (−0.010).
  - Its counterpart without radar, R0+T2+K40, does beat it in both bins: +0.019 and +0.004, each more than its SD.
  - R0+T2+K40 was not chosen by the rule: the rule explored R2 variants, and R0 is the counterpart for the radar
    gain.
  - On val, R0+T2+K40 also loses to the switch: 0.426 against 0.445.
- **Radar gain in the final model, seed against seed, is now negative beyond 50 m:**

  | Split | Distance | Radar gain | Positive seeds |
  |---|---|---|---|
  | select | 50–200 m | −0.002 | 0 of 6 |
  | select | 100–200 m | −0.014 | 0 of 6 |
  | val | 50–200 m | −0.003 | 1 of 6 |
  | val, rain | 50–200 m | −0.022 | 0 of 6 |

  It remains positive at 30–50 m: +0.004 on `select` and +0.006 on val, 6 of 6 on both.
- **`select` does not predict val beyond 50 m.**
  - On `select` the final model beats SECOND beyond 50 m by 0.014. On val it loses by 0.022.
  - The same reversal happened in Phase 1. The main difference is the camera detector: Cascade recall at 100–200 m is
    0.353 on `select` and 0.227 on val (Phase 1).
  - Any change that relies on the camera far away looks better on `select` than it will on val.

**Reproduce** (about 4 h):

```bash
bash results/next_round/phase3_run.sh     # visibility cache, Stage A, rule, Stage B, final 6 seeds on select and val
python results/next_round/phase3_report.py > results/next_round/phase3/report.md
```

**Disk.**
- The visibility caches (`runs/next_round/phase1/cells_in/*_vis.pkl`) take 0.21 GB.
- The epoch 10–30 SECOND checkpoints of the four Phase 1 detectors were deleted (0.76 GB; only epoch 40 is used).
- Free disk is 10 GB.

## Phase 4: gated fusion (2026-11-20)

`fusion.GatedFusion` (tokens G and C in `tune_fusion`) writes the score as
logit(s3D) + g_cam · camera term + g_radar · radar term.
- The camera term is the usual max over a candidate's cells. The radar term comes from a small per-candidate network.
- The two gates are sigmoids of projected 2D height, LiDAR points, radar returns, moving radar returns (|v| ≥ 1 m/s)
  and distance. C adds a one-hot driving condition.
- Both terms start at zero, so training starts from SECOND's ranking.
- Parameters: 2,626 (G) and 2,690 (G+C). Both use K40.
- The rule in PLAN.md includes the camera-removal veto: the candidate is also scored on `select` with every 2D box removed.

**Candidates (select, seeds 1–3, APCD, change from R2+T2+K40; SD is the candidate's).**

| Configuration | 50–200 m | 100–200 m | No camera, 50–200 m | No camera, 100–200 m | Outcome |
|---|---|---|---|---|---|
| R2+T2+K40 (reference) | 0.5783 | 0.3976 | 0.5339 | 0.3722 | |
| **R2+T2+K40+G** | **+0.0074 (SD 0.0018)** | **+0.0039 (SD 0.0022)** | +0.0152 | +0.0100 | **adopted** |
| R2+T2+K40+G+C | −0.0005 | −0.0205 | −0.0071 | −0.0243 | fails |

- G passes in both bins, and the veto does not fire: without the camera, G is better than the reference.
- G does not beat the range switch on `select` at 100–200 m: 0.4015 against 0.405.
- The condition flag hurts. Its gates go to about 0.95 in rain on `select`, where SECOND alone already scores 0.954
  beyond 50 m, and its 100–200 m score falls by 0.02.

**Final, 6 seeds, APCD beyond 50 m (mean ± SD).**

| Model | select 50–200 | select 100–200 | val 50–200 | val 100–200 |
|---|---|---|---|---|
| SECOND alone | 0.564 | 0.405 | 0.446 | 0.284 |
| switch(R2+T2) | 0.562 | 0.405 | 0.445 | 0.284 |
| R0+T2 | 0.574 | 0.386 | 0.398 | 0.248 |
| R2+T2 | 0.574 | 0.377 | 0.399 | 0.243 |
| R2+T2+K40 (Phase 3) | 0.5782 ± 0.0015 | 0.3958 ± 0.0038 | 0.4231 ± 0.0027 | 0.2722 ± 0.0032 |
| R0+T2+K40+G | 0.5831 ± 0.0020 | 0.4084 ± 0.0028 | 0.4224 ± 0.0040 | 0.2744 ± 0.0055 |
| **R2+T2+K40+G** | **0.5856 ± 0.0034** | **0.4020 ± 0.0036** | **0.4199 ± 0.0055** | **0.2613 ± 0.0099** |

- **On val, the gated model is no better than R2+T2+K40:** −0.003 at 50–200 m and −0.011 at 100–200 m. It still loses
  to SECOND and to the switch beyond 50 m, by 0.026 and 0.025.
- **Where it helps on val:**

  | Region | Change from SECOND |
  |---|---|
  | 30–50 m, overall | +0.026 |
  | 30–50 m, rain | +0.090 |
  | 50–100 m, night | +0.019 |

  In rain beyond 100 m it is −0.041.
- **Radar gain, seed against seed (R2+T2+K40+G − R0+T2+K40+G):**

  | Split | Distance | Radar gain | Positive seeds |
  |---|---|---|---|
  | select | 50–200 m | +0.003 | 5 of 6 |
  | select | 100–200 m | −0.006 | 1 of 6 |
  | val | 30–50 m | +0.006 | 6 of 6 |
  | val | 50–200 m | −0.002 | 3 of 6 |
  | val | 100–200 m | −0.013 | 1 of 6 |
  | val, rain | 100–200 m | −0.044 | 0 of 6 |

**Gates** (`figures/gates.png` for val; `figures/gates_select.png` shows both variants on `select`):
- **g_radar falls with distance in every condition.** On val, for true cars, it goes from 0.69–0.80 at 0–30 m to
  0.36–0.39 at 100–200 m. The network learned to trust radar less far away. That matches the radar gain, which is
  negative beyond 100 m.
- **g_cam barely moves.** For true cars it is 0.48–0.69 everywhere. For other candidates it is about 0.4–0.55.
- **Without the condition flag, the gates hardly separate conditions.** The spread between conditions is at most
  about 0.1 in any bin. With the flag, they separate by condition and overfit `select`.

**Reproduce** (about 1 h 45 min):

```bash
bash results/next_round/phase4_run.sh     # moving-return cache, candidates, no-camera scoring, rule, final 6 seeds, gate plots
python results/next_round/phase4_report.py > results/next_round/phase4/report.md
```

**Disk.**
- The moving-return caches (`runs/next_round/phase1/cells_in/*_moving_*.pkl`) take 0.10 GB.
- Free disk is 9.9 GB.

## Phase 5: the ceiling of a radar detector (2026-11-20)

`clocs_r/oracle_inject.py` (rule in PLAN.md).
- **Target cars:** scored cars beyond 50 m with no LiDAR point in the box and at least one radar return in the gate
  (3 sweeps, δ0 0.5 m, κ 0.02).
- **Injection:** each target's ground-truth box is added as an extra candidate with s3D = 0.1, with all the inputs a
  SECOND candidate has.
- **Variants:**
  - **low:** each model rescores the injected candidates;
  - **perfect:** their final score is set to 1.
- **Models:** all from saved weights, seeds 1–6, nothing trained.
- **Check:** scored without injection, every model reproduces its stored results; the largest difference is 3e-5 APCD.
- Full tables: `phase5/report.md`.

**Target cars beyond 50 m.**

| Split | Scored cars | No LiDAR point | Targets (no LiDAR + radar) | Targets SECOND already finds (score ≥ 0.05 within 2 m) |
|---|---|---|---|---|
| select | 9,634 | 1,045 (10.8%) | 384 (4.0%) | 209 |
| val | 16,372 | 1,777 (10.9%) | 862 (5.3%) | 297 |

- **Most targets are urban beyond 100 m:** 158 on `select` and 482 on val.
- **On val, night and rain have the highest target rates** beyond 50 m: 7.3% (99 cars) and 9.8% (131 cars).
- **`select` has none in rain and 23 at night,** so it under-represents exactly the conditions where radar should
  matter.

**Ceiling, APCD at 50–200 m (mean of 6 seeds; every seed positive for every model).**

| Model | select, none | select, low | select, perfect | val, none | val, low | val, perfect |
|---|---|---|---|---|---|---|
| SECOND alone | 0.564 | +0.010 | +0.039 | 0.446 | +0.023 | +0.055 |
| switch(R2+T2) | 0.562 | +0.010 | +0.039 | 0.445 | +0.023 | +0.055 |
| R0+T2 | 0.574 | +0.005 | +0.036 | 0.398 | +0.017 | +0.059 |
| R2+T2 | 0.574 | +0.005 | +0.036 | 0.399 | +0.016 | +0.059 |
| R2+T2+K40 | 0.578 | +0.007 | +0.036 | 0.423 | +0.020 | +0.058 |
| R0+T2+K40+G | 0.583 | +0.008 | +0.036 | 0.422 | +0.020 | +0.057 |
| **R2+T2+K40+G** | **0.586** | **+0.007** | +0.036 | 0.420 | +0.019 | +0.058 |

- **The rule says no.** The low ceiling of R2+T2+K40+G at 50–200 m on `select` is +0.0070, against a threshold of
  +0.010.
- **val says the ceiling is about 2.7× larger:** +0.019 low and +0.058 perfect. That is because val has the night and
  rain targets `select` lacks.
- **The gains on val with R2+T2+K40+G, low variant:**

  | Region | Low ceiling |
  |---|---|
  | urban, 100–200 m | +0.042 |
  | night, 50–100 m | +0.066 |
  | rain, 100–200 m | +0.038 |
  | highway, 50–200 m | +0.002 |

- **The fusion models give injected candidates less than SECOND does** (about 0.075 instead of 0.1). Their low
  ceiling is therefore below SECOND's on both splits.
- **Even with perfect injection, R2+T2+K40+G stays below SECOND with perfect injection** on val: 0.477 against 0.500.
- **NMS keeps 237 of 384 injected candidates on `select` and 695 of 862 on val** (low variant). Most of the rest
  overlap a SECOND box with a higher score.
- **Caveats on the perfect ceiling:**
  - It also counts cars SECOND already finds but places imprecisely, because the exact box replaces SECOND's box.
  - The gate widens with range (3.5 m beyond the box at 150 m), so some targets may only have returns from a neighbor.

**Reproduce** (about 3 h 15 min):

```bash
bash results/next_round/phase5_run.sh     # injection caches, then every model with and without the injected cars
python results/next_round/phase5_report.py > results/next_round/phase5/report.md
```

**Disk.**
- The injection caches (`runs/next_round/phase1/cells_in/*_oracle_*.pkl`) are a few MB.
- Free disk is 9.8 GB.

## Phase 6: a radar detector as a second candidate source (2026-11-20)

Run on the user's decision; the Phase 5 rule had said no (addendum in PLAN.md). Full tables: `phase6/report.md`.

**Code:**
- `clocs_r/radar_dataset.py`: points, labels and the pillar scatter fix below.
- `clocs_r/configs/pointpillar_radar_aimotive.yaml`.
- `train_second.py --config` and `export_3d.py --config --nms`.
- `clocs_r/radar_candidates.py` and token RD in `tune_fusion`.
- `results/next_round/phase6_extra.py`.

**The radar detector.**
- Radar-only PointPillars, 4.8M parameters, trained on `fit` for 40 epochs (2 h on the A100 slice).
- Inputs: 5 radar sweeps, about 106 returns per frame. Only about 21% of car boxes have a radar return in the gate,
  so only those are training labels.
- **Fix:** OpenPCDet's `PointPillarScatter` takes the batch size from the frames that have pillars. A batch ending in
  frames with no radar point in range crashed the export. `radar_dataset.RadarPillarScatter` takes it from the batch
  instead, with identical output otherwise (tested). It has no parameters, so the trained weights are unaffected.

| APCD | 50–100 m | 100–200 m | 50–200 m |
|---|---|---|---|
| radar detector, select | 0.237 | 0.094 | 0.199 |
| SECOND, select | 0.628 | 0.405 | 0.564 |
| radar detector, val | 0.202 | 0.044 | 0.126 |
| SECOND, val | 0.599 | 0.284 | 0.446 |

Even at night on val, its best condition, the detector scores 0.218 beyond 50 m against SECOND's 0.428. AP3D@0.5 beyond
50 m is 0.008 on `select` and 0.002 on val.

**Candidates.**
- Radar detections with score ≥ 0.05 and no SECOND candidate within 2 m become new candidates: about 17 per frame.
- Only 2.0% of them on `fit`, 0.7% on `select` and 2.0% on val are within 2 m of a scored car.

**Rule (select, seeds 1–3).** R2+T2+K40+G+RD scores 0.483 at 50–200 m and 0.264 at 100–200 m, against the
reference's 0.586 and 0.402. It fails by −0.103 and −0.137, so **R2+T2+K40+G stays the current model.**

**Final, 6 seeds, APCD.**

| Model | select 50–200 | select 100–200 | val 50–200 | val 100–200 |
|---|---|---|---|---|
| SECOND alone | 0.564 | 0.405 | 0.446 | 0.284 |
| switch(R2+T2) | 0.562 | 0.405 | 0.445 | 0.284 |
| R0+T2+K40+G | 0.583 | 0.408 | 0.422 | 0.274 |
| R2+T2+K40+G | 0.586 | 0.402 | 0.420 | 0.261 |
| **R2+T2+K40+G+RD** | **0.481 ± 0.004** | **0.261 ± 0.006** | **0.303 ± 0.002** | **0.116 ± 0.002** |

- **Radar gain** (RD − R0+T2+K40+G, seed against seed), beyond 50 m: −0.102 on `select` and −0.119 on val, 0 of 6
  seeds positive. Every condition and far bin is negative.
- **The only gain** is at 30–50 m on val: +0.014 overall and +0.023 urban, 6 of 6.

**Where the loss comes from** (extras; "none" drops the radar-sourced candidates, "union" keeps them):
- **The `rd` input does no harm.**
  - R2+T2+K40+G+RD without its radar-sourced candidates matches R2+T2+K40+G: 0.585 against 0.586 on `select`, 0.421
    against 0.420 on val.
  - All of the loss comes from the new candidates.
- **They add recall, but at about 3–5 false alarms per frame.**

  | Split | Recall beyond 50 m (score ≥ 0.1), without → with | Radar-sourced boxes kept per frame | True |
  |---|---|---|---|
  | select | 0.662 → 0.672 | 5.4 | 1.3% |
  | val | 0.408 → 0.441 | 3.6 | 5.1% |

  - The false alarms outrank real far cars, so APCD drops.
- **SECOND ∪ radar shows the same pattern:**

  | Split | APCD 50–200 m | Recall beyond 50 m |
  |---|---|---|
  | val, SECOND alone | 0.446 | 0.504 |
  | val, SECOND ∪ radar | 0.377 | 0.535 |

  - Recall rises most at night on val: from 0.483 to 0.540.
- **Reproduction:** every Phase 4 model and SECOND, scored on the same frames without the radar-sourced candidates,
  reproduces its stored result exactly.

**Reproduce** (about 4 h 45 min):

```bash
bash runs/next_round/phase6/train_radar.sh     # radar PointPillars on fit (2 h)
bash results/next_round/phase6_run.sh          # export, candidates, detector alone, rule, final 6 seeds, extras
python results/next_round/phase6_report.py > results/next_round/phase6/report.md
```

**Run notes.**
- The first fusion run stopped on a CUDA "unspecified launch failure" at the first step of seed 1. The kernel log
  showed no GPU error, and the rerun was clean.
- **Disk:** the radar candidate caches in `runs/next_round/phase1/cells_in/*_rd_*.pkl` take 0.26 GB, and
  `runs/next_round/phase6` takes 0.17 GB. Free disk is 9.0 GB.

## Phase 6b: radar-only candidates precise enough to help (2026-11-20)

Pre-registered in PLAN.md (Phase 6b and its addenda). Phase 6 showed that radar-only candidates add recall but are
95–99% false, so Phase 6b tried to make them precise.

**Code:**
- `clocs_r/radar_clutter.py`: the per-point clutter classifier.
- `clocs_r/radar_filters.py`: precision filters and fast APCD.
- `clocs_r/radar_dataset.py`: Doppler motion correction (`motion=True`) and the clutter probability as a 7th point
  feature.

**Per-point clutter classifier** (`phase6b/clutter.json`):
- A small MLP over each radar return's 6 features, its range and neighbourhood statistics within 2 m.
- Label: inside a car box enlarged by 0.5 m.
- Separation of car returns from clutter:

  | Split | AUC, all | AUC, beyond 50 m | Car returns |
  |---|---|---|---|
  | `fit` (training) | 0.955 | 0.933 | 7.6% |
  | `select` | 0.932 | 0.901 | 11.6% |

**The filter audit on D1** (`phase6b/d1_select.json`, `select`):
- 25 filters: score thresholds × cues (none, a moving return, a LiDAR point, persistence across frames, moving and
  persistent).
- Two merge modes: below every base box, or scored by the filter's precision on `fit`.
- Both SECOND and R2+T2+K40+G were used as bases.
- **No filter helped.** The best gave +0.00001 APCD at 50–200 m on top of R2+T2+K40+G. Every filter's precision beyond
  50 m was below about 3%.

**Closed after D1** (the user's re-evaluation, before any D2 or D3 result; PLAN.md addendum, REEVALUATION.md):
- Radar never creates candidates in the CLOCs version.
- The filter audit was not run for D2 and D3, and val was not scored for Phase 6b.
- D2 and RadarDistill (D3) were still trained; their boxes are a CLOCs sensor in Phase 7.

**Reproduce:**

```bash
python -m clocs_r.radar_clutter --out runs/next_round/phase6b/clutter.pt --report results/next_round/phase6b/clutter.json
python -m clocs_r.radar_filters --dets runs/next_round/phase6/radar_dets --tag d1 --out results/next_round/phase6b/d1_select.json
```

## What each addition contributes (2026-11-20 to 2026-11-21)

Every addition so far is compared with the same model without it, seed against seed (6 seeds), on `select` and val.
The full table is in [contributions.md](contributions.md) (`contributions.py`, runs in `runs/next_round/contrib_run.sh`).

| Addition | Val 50–200 m | Positive seeds | Verdict |
|---|---|---|---|
| The camera-LiDAR fusion network (R0+T2) vs SECOND alone | −0.047 | 0/6 | Fusion loses beyond 50 m on val, though it helps within 50 m (+0.019 at 0–30 m) |
| Residual scoring T2 (no radar) | +0.029 | 6/6 | Essential |
| Top-40 2D boxes K40 | +0.024 | 6/6 | Helps |
| The camera in the gated model | +0.010 | 5/6 | Helps; mostly within 50 m (+0.030 at 0–30 m) |
| Far-range loss weight 2 vs 1 | +0.002 | 4/6 | Makes no difference |
| Radar Doppler feature R2, plain fusion | +0.000 | 3/6 | Helps only at 30–50 m (+0.007, 6/6) |
| Gated model G | −0.003 | 1/6 | Does not help |
| Range switch at 50 m vs SECOND | −0.000 | 0/6 | SECOND alone beyond 50 m by construction |
| Radar-detector candidates RD | −0.117 | 0/6 | Harmful |

## Phase 7: radar through a detector, and a situational selector (2026-11-20 to 2026-11-21)

Pre-registered in PLAN.md (Phase 7 and its addenda). Decisions, findings and their reasons are in DECISIONS.md
(2026-11-21). The full tables, conditions and gate weights are in [phase7/report.md](phase7/report.md).

**The question:** does radar help LiDAR beyond 50 m?
- Radar enters each model by one route only: point features or a radar detector's boxes.
- Radar never creates candidates.
- Camera-free models isolate radar's contribution.

**Models.**

| Label | Model |
|---|---|
| L | SECOND alone |
| L0 | R0+T2+NC: CLOCs with neither camera nor radar (the control) |
| LRf | R2+T2+NC: radar as point features |
| LRd | R0+T2+RP: radar through D2's boxes as CLOCs pairs (cells: affinity 1 − d/2 m, radar score) |
| LC | R0+T2+K40: camera CLOCs |
| S | A selector weighting L, LC and LRd per candidate from its situation; trained on `fit` with out-of-fold expert logits |
| B | A fixed blend of the same three experts, the same for every candidate (post hoc) |
| inc | R2+T2+K40+G, the incumbent |

**Code:**
- `clocs_r/radar_pairs.py`: token RP.
- Token NC in `tune_fusion`.
- `clocs_r/selector.py` (with `--blend`).
- `clocs_r/radardistill_aimotive.py` with `clocs_r/configs/radardistill/`.
- `clocs_r/bootstrap.py`: the scene bootstrap.
- `clocs_r/selector_ceiling.py`.

**Radar detectors alone** (`select`, APCD at 50–200 m; SECOND 0.564):

| Detector | APCD |
|---|---|
| D1: Phase 6 PointPillars | 0.199 |
| D2: motion correction and clutter probability | **0.217** |
| D3: RadarDistill, the authors' code (teacher 10 epochs, student 20) | 0.215 |

D2 was chosen by the rule.

**Results** (APCD, 6 seeds):

| Model | `select` 50–200 | `select` 100–200 | val 50–200 | val 100–200 |
|---|---|---|---|---|
| L | 0.5639 | 0.4052 | **0.4455** | 0.2841 |
| L0 | 0.5605 | 0.4062 | 0.4282 | 0.2729 |
| LRf | 0.5729 | 0.4138 | 0.4233 | 0.2731 |
| LRd | 0.5753 | 0.4162 | 0.4415 | 0.2858 |
| LC | 0.5807 | 0.4096 | 0.4264 | 0.2789 |
| S | 0.5928 | 0.4184 | 0.4407 | 0.2877 |
| B (post hoc) | **0.5933** | **0.4261** | 0.4428 | **0.2912** |
| inc | 0.5856 | 0.4020 | 0.4199 | 0.2613 |

**Scene-bootstrap 95% intervals** (`select` 24 scenes, val 25):

| Comparison | `select` 50–200 | val 50–200 | val 100–200 |
|---|---|---|---|
| LRd − L0 | +0.0148 [+0.0070, +0.0218] | +0.0133 [+0.0049, +0.0228] | +0.0129 [+0.0078, +0.0179] |
| LRf − L0 | +0.0124 [−0.0018, +0.0262] | −0.0049 [−0.0128, +0.0039] | +0.0003 [−0.0065, +0.0060] |
| LRd − LRf | +0.0024 [−0.0078, +0.0128] | +0.0182 [+0.0104, +0.0238] | +0.0126 [+0.0068, +0.0191] |
| LRd − L | +0.0114 [+0.0032, +0.0185] | −0.0040 [−0.0146, +0.0050] | +0.0016 [−0.0042, +0.0079] |
| S − B | −0.0005 [−0.0040, +0.0038] | −0.0022 [−0.0052, +0.0018] | −0.0035 [−0.0076, +0.0015] |
| B − L | +0.0294 [+0.0121, +0.0480] | −0.0027 [−0.0130, +0.0057] | +0.0070 [−0.0057, +0.0199] |

**Findings:**
1. **Radar through a detector helps CLOCs beyond 50 m on both splits.** LRd − L0 is above zero in both far bins on
   `select` and on val, with 6/6 seeds each time.
2. **Radar as point features does not.** Its `select` gain vanished on val, and on val the detector route is clearly
   better (LRd − LRf).
3. **On val, no fusion model beats SECOND alone in the pooled 50–200 m bin.**
   - The camera-free CLOCs control L0 is about 0.017 below SECOND. Radar recovers most of that.
   - Per bin, B is at or above SECOND everywhere on val: +0.021 at 0–30 m, +0.019 at 30–50 m, −0.001 at 50–100 m and
     +0.007 at 100–200 m. It loses only in the pooled bins, so its scores are less comparable across ranges than
     SECOND's.
   - Val's 50–200 m bin is 50% cars at 100–200 m, against 29% on `select`.
4. **The situational selector adds nothing beyond a fixed blend.**
   - B (L 0.00, LC 0.72, LRd 0.28) matches or beats S on both splits.
   - The oracle ceiling on `select` puts a perfect per-distance blend only +0.0025 above B at 100–200 m, below the
     pre-registered 0.005, so selector work stopped.
5. **By the pre-registered rule,** S(R0+T2+RP) replaces the incumbent on `select`. B is not eligible, because it was
   defined after the `select` results.

**Val exposure:** val was scored once in Phase 7 (2026-11-21), after every model was frozen. No setting was changed
after it.

**Reproduce:**

```bash
bash results/next_round/phase7_run.sh      # radar detectors, experts, selector, the rule (select only)
bash results/next_round/phase7_val.sh      # B on select, val once for every model, scene bootstrap
python -W ignore -m clocs_r.selector_ceiling --radar_dets runs/next_round/phase7/d2_dets --out results/next_round/phase7/ceiling.json
python results/next_round/phase7_report.py > results/next_round/phase7/report.md
```

**Run notes:**
- **CUDA "unspecified launch failure"** on the shared vGPU, at the first backward step of some workers. Mitigated with
  a staggered start (`STAGGER_S` = 20 s) and up to 3 retries per step.
- **`tune_fusion` overwrote runs** when seed groups were run separately. It now merges by seed, and seeds 1–3 were
  re-scored from their weights (seed 1 reproduced exactly).
- **Two changes to RadarDistill's code** (PLAN.md addenda): the deformable convolution uses torchvision's op, and the
  radar feature count comes from the dataset instead of a hard-coded 6. The export skips batches with no radar point
  in range.
- **Disk:** 7.3 GB free.
