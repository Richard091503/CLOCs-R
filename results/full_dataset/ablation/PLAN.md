# Full-dataset ablation plan

Written on 2026-11-15, before any ablation run. This repeats the repository's earlier ablation process on the full dataset (all four conditions, every frame, range-weighted training). The earlier process was gate selection, then a feature-set ablation, then fusion tuning ([`../../clocs_radar/`](../../clocs_radar/README.md)). **Its outcome determines the final CLOCs and CLOCs-R configurations.**

**Disclosure.** Val has already been read for R0, R0+T2, R2, R3, R2+T2 and R3+T2 (first full-dataset round, [`../README.md`](../README.md)). Nothing below reads val before step 4. Holdout and val stay as they are.

## Held fixed

- **Detectors:** SECOND (`runs/second/checkpoint_epoch_40.pth`) and Cascade R-CNN (`runs/cascade/epoch_4.pth`), with their exported candidates.
- **Fusion network and training:** the same network, 3 epochs, seeds 1, 2 and 3, and the range-weighted fusion loss (far weight 2).
- **Selection metric:** center-distance AP beyond 50 m (`overall`, bin `50-200`) on **holdout**, mean of the 3 seeds.
- **Reuse:** a run already trained with identical settings, seeds and gate in the first round is reused rather than retrained (its `settings`, `far_weight` and `gate` must match).

## 1. Radar gate

- **Model:** R3 (support and Doppler speed, the paper's proposed set), 3 sweeps.
- **Grid:** δ0 ∈ {0.25, 0.5, 1.0} m × κ ∈ {0.01, 0.02, 0.04}, the same 9 gates as before.
- **Rule:** the highest mean wins.

## 2. Feature sets

- **Configurations:** R0 (no radar) and R1–R7 of Table II of the paper, all at the selected gate:
  - R1: support
  - R2: Doppler speed
  - R3: support and Doppler speed
  - R4: R3 plus Doppler spread
  - R5: support, range residual Δ and offset o
  - R6: R3 plus Δ
  - R7: all five features
- **Rule:** the highest mean among **R1–R7** is the selected set S. R0 is the reference.

## 3. Fusion changes, each on top of S

| Change | What it does |
|---|---|
| T1 far targets | Beyond 50 m, a candidate counts as positive at BEV IoU ≥ 0.5 and as negative at ≤ 0.3 (0.7 and 0.5 elsewhere) |
| T2 residual | Fused logit = SECOND's logit + network output; the last layer starts at zero |
| T3 logit input | SECOND's score enters as logit / 5 instead of a probability |
| T4a 5 sweeps | M = 5, each return moved along its line of sight by its Doppler speed times its age |
| T4b 10 sweeps | The same with M = 10 |
| T5 RCS | Median log radar cross-section of the gated returns, as an extra channel |

**Decision rule:**

1. Each change is tested alone as S+Tx.
2. **Adoption:** a change is adopted if its mean exceeds S's mean by more than the standard deviation of S's 3 seeds (sample standard deviation, ddof = 1).
3. **Combination:** if two or more changes are adopted, their combination is also trained. T4a and T4b are alternatives, so only the better of the two can enter.
4. **Final CLOCs-R:** the highest mean among S, the adopted single changes and the combination.
5. **Final CLOCs:** the radar-free twin of the final CLOCs-R. That is R0 with the same fusion changes, minus the radar-only ones (T4a, T4b and T5).

## 4. Val, once

- **What is scored:** every network trained in steps 2–3, plus the twin, is scored on holdout and val.
- **More seeds:** the final CLOCs-R and the final CLOCs also get seeds 4, 5 and 6, trained identically. The val report uses all 6 seeds for these two. Selection used seeds 1–3 only.
- **Radar gain:** the final CLOCs-R minus the final CLOCs, seed against seed. It is reported by condition (overall, clear, adverse, highway, urban, night, rain, without tunnels), by distance bin, and per scene.
- **Camera removed:** the final CLOCs-R and CLOCs are scored with every 2D candidate removed at test time. As in the earlier ablation, the network rarely sees this input during training, so this is evidence about the camera's role, not a robustness result.
- **Latency:** radar-feature time and fusion forward-pass time per frame for R0–R7 at the selected gate.

Holdout has no rain cars beyond 50 m and only 13 night cars at 100–200 m (see [`../DATASET.md`](../DATASET.md)). So this selection cannot reward performance in far rain or far night, and the val results for those conditions are out-of-sample.

Pipeline: `python -m clocs_r.ablation_full` (resumable; each stage skips work whose output exists). Outputs: `results/full_dataset/ablation/` and `runs/ablation/`.

## Notes added while running (2026-11-15)

- **Seeds 4–6 were reused, not retrained.** Seeds 4–6 of R2+T2 and R0+T2 had already been trained for the R3 rerun (`../rerun_r2.json`, `../rerun_r3.json`), with identical settings, far weight 2 and the selected gate. Under the reuse rule above they were copied into `extra_seeds.json` rather than retrained. Their val numbers had already been read, which is part of the disclosure above (R0+T2 and R2+T2).
- **A scoring bug was fixed mid-run.** Step 4 stopped when it re-scored R2+T1 from its saved weights, because the far-target labels were rebuilt from training data that scoring does not load. Scoring no longer builds them, since they only matter for training. The 9 networks already scored were kept, and step 4 resumed. No selection decision was affected: steps 1–3 had finished, and their outputs were unchanged.
