# Where the next round stands (handoff note; update it whenever the state changes)

**Last updated: 2026-11-23, 16:00 UTC** (disk cleanup; the sparse version is on branch `sparse`, `results/sparse/STATUS.md`).

**Branches (2026-11-22, `DECISIONS.md` item 15):**
- `master` is the LCR class-project version: `clocs_r/run_lcr.sh`, `results/lcr/` and `docs/MODELS.md`.
- `thesis` is this full record.
- The sparse LiDAR + radar version (`V2_PLAN.md`) is next, on its own branch from `thesis`.

**Run outputs on the VM:** the `master` checkout's `runs/` (shared by every branch). `runs/lcr/` is links into
`runs/next_round/`.

**Disk cleanup, 2026-11-23** (user's request: free disk, with every result kept and no model needing retraining).
Free disk went from 6.4 GB to 17 GB. Every number of every model stays in git (this branch's `results/`), and every
trained network is still on disk.

| What | Freed | What was done | How to get it back |
|---|---|---|---|
| Base CLOCs cells: `phase1/cells_in/{fit,select,val}_base.pkl` | 7.2 GB | Deleted. A test rebuild of `select` was byte-identical to the stored file and took 29 s. | `python -m clocs_r.cache_cells --split select,val,fit --no_radar --out runs/next_round/phase1/cells_in --cand3d runs/next_round/phase1/cand_in/3d --cand2d runs/next_round/phase1/cand_in/2d`, about 5 min. On `master`, `run_lcr.sh` rebuilds them by itself. |
| Optimizer state in 9 finished OpenPCDet checkpoints: SECOND (`fit`, fold 0–2, the earlier `runs/second`), D1, D2, and D3's teacher and student | 0.60 GB | Removed. The weights were checked bit-identical, and each file now carries a `note` key. | Not needed: export and scoring load only `model_state`. Training can no longer resume from these files; a rerun of training needs a fresh output folder. |
| D1's and D3's exported boxes: `phase6/radar_dets`, `phase7/d3_dets` | 0.21 GB | Deleted (results in `phase6/radar_detector.json`, `phase6b/d1_select.json`, `phase7/radar_detector_d3.json`). | Re-export from the kept weights with the export lines of `phase6_run.sh` (`export_3d ... --nms`) and `phase7_run.sh` (`radardistill_aimotive --stage export`). |
| Not project data | 2.5 GB | Deleted three old Claude Code VS Code extension versions (2.1.327–2.1.329), VS Code's cached extension downloads, the installed VS Code `.deb` in `~/Downloads`, Firefox's cache, and conda's unused package cache. | Not needed. |

**Kept on purpose:**
- every final detector network, including the Cascade R-CNNs (`fit`, fold 0–2, the earlier `runs/cascade`; weights
  only already);
- the candidates (`phase1/cand_in`, 1.7 GB), from which the cells are rebuilt and SECOND is scored;
- the cross-fitting detectors (`det/fold0`–`fold2`), needed to test the item 12 hypothesis;
- D2's boxes (`phase7/d2_dets`), used by LCR;
- the smaller radar, visibility and pairing caches in `cells_in` (1.3 GB);
- `~/.cache/torch`, the COCO and ImageNet weights that Cascade R-CNN training starts from.

**Main model of the CLOCs version: LCR** (`R0+T2+K40+RC`), chosen by the user over the rule's B (`DECISIONS.md` item 13).
- Val is scored: 0.4271 at 50–200 m and 0.2779 at 100–200 m. Both are below SECOND alone (0.4455 and 0.2841).
- **The LCR run is DONE** (00:18 UTC), including the bootstrap (`DECISIONS.md` item 13).
- **LCR-D is DONE** (04:01 UTC). The rule keeps LCR, and every LCR-D − LCR interval includes 0 (`DECISIONS.md` item 14). It ran (`R0+T2+K40+RC+DF`, `results/next_round/phase7_lcrd.sh`, launched 22:25 UTC; log
  `runs/next_round/phase7/lcrd.log`; `DECISIONS.md` item 14).
  - It waits for "LCR DONE", then trains 6 seeds, applies the rule (`decision_lcrd.json`), scores val once and runs the
    bootstrap (`bootstrap_lcrd_{select,val}.json`).
  - Restart: rerun the script.
- The cut-down around LCR is done (`DECISIONS.md` item 15).

## Running now

Nothing. **The ceiling check is DONE** (19:38 UTC). The distance oracle minus B is +0.0025 at 100–200 m, below 0.005, so the selector variants are not run (`DECISIONS.md` item 10). **Step 2 is DONE (20:30 UTC):**
- `phase7/report.md` (`phase7_report.py`) and the figure `figures/phase7_weights.png`;
- README sections for Phase 6b, the contributions and Phase 7;
- a correction to `DECISIONS.md` item 3: the selector's weights were first quoted from seed 1 only, and the 6-seed
  means are milder.

**Running now: LCR** (`results/next_round/phase7_lcr.sh`, started 20:32 UTC; log `runs/next_round/phase7/lcr.log`).
- What it is (the user's choice for a simple final model): one CLOCs network over SECOND, Cascade R-CNN and D2,
  `R0+T2+K40+RC`, token RC in `radar_pairs.combined_cells`.
- Pre-registered in the PLAN.md LCR addendum, as post-val development.
- Steps: 6 seeds on `fit`, then the rule on `select` (LCR is final if not below B by more than its seed SD in both far
  bins), then val once, then the bootstrap (LCR − LC, LCR − LRd, LCR − B, LCR − L).
- Outputs: `phase7/lcr.json`, `decision_lcr.json`, `lcr_val.json`, `bootstrap_lcr_{select,val}.json`.
- Restart: rerun the script. Training merges by seed, so redoing a seed group is safe.
- **Disk:** 6.7 GB free. The 0.6 GB drop on 2026-11-21 was VS Code updating the Claude Code extension, not the
  project. Earlier today, `results/next_round/phase7_val.sh` ran from 11:42 to 18:58 UTC, with the user's go-ahead. Log:
`runs/next_round/phase7/val.log`; step logs are `blend.log`, `val_fusion.log`, `val_selector.log` and `bootstrap.log`
in the same folder.
1. The post-hoc fixed-blend ablation B (PLAN.md, Phase 7 blend addendum) is trained on `fit` and scored on `select`,
   about 70 min. Output `phase7/blend.json`; gates `phase7/blend_R0+T2+RP_s{1-6}.pth`.
2. Val is scored once for L0, LRf, LRd, S and B, from frozen weights.
3. The scene bootstrap runs on `select` and val.

**Step 1 is done (12:50 UTC). B on `select`** (6 seeds; S − B paired by seed):

| Bin | S | B | S − B | S − B > 0 |
|---|---|---|---|---|
| 50–200 m | 0.5928 | 0.5933 | −0.0005 | 1/6 seeds |
| 100–200 m | 0.4184 | 0.4261 | −0.0077 | 0/6 seeds |

- S does not beat B by the pre-registered rule. So S's gain is explained by a fixed blend, not by choosing by
  situation.
- B's learned weights are the same for every seed: L 0.00, LC 0.72, LRd 0.28.
- B is ahead at 100–200 m in every condition (0/6 seeds favour S). S is ahead by about +0.001 below 100 m.

**Val, 50–200 m** (6 seeds; preliminary until the bootstrap finishes):

| Model | APCD |
|---|---|
| L0 | 0.4282 |
| LRf | 0.4233 |
| LRd | 0.4415 |
| S | 0.4407 |
| B | 0.4428 |

- LRd − L0 = +0.013, ahead on all 6 seeds.
- LRf − L0 = −0.005, behind on all 6 seeds.
- The best single expert on `select` was LC.

**`phase7_val.sh` is DONE** (2026-11-21 18:58 UTC). The bootstrap results and their interpretation are in
`DECISIONS.md` (2026-11-21, item 8). The headline is that on val no fusion model beats SECOND alone beyond 50 m.

**Running now:** `runs/next_round/phase7/ceiling_after_val.sh`, started 18:59 UTC (waits for "PHASE 7 val DONE"; log
`runs/next_round/phase7/ceiling.log`).
- It runs `clocs_r.selector_ceiling`, the oracle ceiling of situation-dependent blends on `select` (PLAN.md ceiling
  addendum), into `phase7/ceiling.json`.
- Decision rule: the two selector variants run only if the distance oracle minus B ≥ 0.005 at 100–200 m.

**Plan approved by the user (2026-11-21), after the ceiling check:**
1. The two variants, only if the rule allows: S plus radar-detector evidence, and the selector over the three
   detectors' own scores (D2, and RadarDistill as a named comparison).
2. The Phase 7 report.
3. Cut the CLOCs version down to what earned its place.
4. Then the sparse LiDAR + radar version (`V2_PLAN.md`).

If it fails, the script can be rerun. It skips B's training when `blend.json` already holds 6 runs, and every step is
retried 3 times.

**Phase 7 on `select`** (APCD, 6 seeds; `results/next_round/phase7/`):

| Model | 50–200 m | 100–200 m |
|---|---|---|
| L0 (`R0+T2+NC`, control) | 0.5605 | 0.4062 |
| LRf (`R2+T2+NC`) | 0.5729 | 0.4138 |
| LRd (`R0+T2+RP`, D2's boxes) | 0.5753 | 0.4162 |
| **S (selector over L, LC, LRd)** | **0.5928** | **0.4184** |
| Incumbent R2+T2+K40+G | 0.5856 | 0.4020 |
| LC (`R0+T2+K40`) | 0.581 | 0.410 |
| SECOND | 0.564 | 0.405 |

- **Rule:** S beats the incumbent in both bins, so the current model is S(R0+T2+RP), recorded in
  `phase7/decision.json`.
- **Radar detectors** (alone, 50–200 m): D1 0.199, D2 0.217, D3 0.215. D2 was chosen for RP.
- **The gate weights barely vary with distance:** L 0.01–0.10, LC 0.60–0.70, LRd 0.24–0.35. Part of S's gain may be
  ensembling. The user approved a fixed-blend ablation B: post hoc, labelled as such, and registered in PLAN.md before
  val. S beats B only by the Phase 7 rule; B cannot become the current model.
- **Code fixes during Phase 7:**
  - a staggered CUDA start and retries;
  - seed groups merged by seed in `tune_fusion`;
  - RadarDistill's hard-coded radar feature count (PLAN addendum);
  - the export skips batches with no radar point.

## Still to do

1. **Val scoring and the bootstrap: running** (see above).
2. **The scene-bootstrap tool is written:** `clocs_r/bootstrap.py`, tested in `test_core.py` and smoke-tested on
   real data.
   - Run it on `select` and val with models SECOND, L0, LC, LRf, LRd, S, B and the incumbent.
   - Comparisons: RQ1 (LRf − L0, LRd − L0, LRf − L, LRd − L), RQ2 (LRd − LRf), RQ3 (S − best single expert, S − L,
     S − incumbent), and the blend (S − B, B − best, B − L, B − incumbent).
   - Check that its point estimates equal the stored APCD.
3. **The Phase 7 report and README section:** done (2026-11-21).
4. **Branches** (only when the user says the version is finished; the user pushes):
   - `master` stays at 688b38e;
   - this next-round work, the detector-plus-selector version, goes on its own branch;
   - the sparse LiDAR + radar version gets its own branch, starting from `V2_PLAN.md`;
   - use a git worktree so running jobs are not disturbed.
   - **Never commit or push without the user's explicit request.**

## Where things are

| What | Where |
|---|---|
| Rules | `PLAN.md` (Phases 0–7, dated addenda) |
| Audit | `REEVALUATION.md` |
| Decisions | `DECISIONS.md` |
| The other version | `V2_PLAN.md` |
| Results per phase | `phaseN/report.md`, `README.md` |
| What each addition contributes | `contributions.md` |
| RadarDistill (authors' code, commit a81d248) | `~/RadarDistill`, ops built in place. Changes: the deformable convolution uses torchvision's op; stand-ins for unused devkits. torch-scatter installed in the `clocs` environment. |

**User preferences:**
- Report when a step finishes or something changes, not on a timer.
- Times in US Eastern (UTC−5).
- One heavy job at a time.
- Tell the user before free disk would fall below 10 GB; it is at 8.8 GB, and the user knows.
