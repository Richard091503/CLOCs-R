# Full-dataset ablation

This repeats the repository's ablation (gate, then feature sets, then fusion changes) on the full aiMotive dataset, with range-weighted training. The rules were fixed in [PLAN.md](PLAN.md) before any run. Selection read **holdout only**. Val was scored once, after the final configurations were fixed. The runs finished on 2026-11-16, and every number below is in [tables.md](tables.md), produced by `make_tables.py`.

The metric is center-distance AP on cars beyond 50 m, as the mean of seeds 1–3 for selection. *Radar gain* is CLOCs-R minus its radar-free twin, seed against seed.

## Outcome

| | Configuration | What it is |
|---|---|---|
| **Final CLOCs-R** | **R2+T2** | Doppler speed of the gated radar returns, fused logit = SECOND's logit + network output |
| **Final CLOCs** | **R0+T2** | The same network without radar |
| Radar gate | δ0 = 0.5 m, κ = 0.02, 3 sweeps | The default gate |

These are the same configurations the first full-dataset round selected ([`../README.md`](../README.md)). The second round tested 9 gates, 8 feature sets and 6 fusion changes, and none of the alternatives did better on holdout.

## What each step found

### 1. The gate barely matters

Trained with R3, all 9 gates scored within 0.0044 of each other on holdout (0.4178 to 0.4222). That is smaller than the spread between seeds of a single gate, which is up to 0.018. The default gate came first by 0.0002, so the choice is close to arbitrary. The only weak trend is that the narrowest gate with the steepest growth (δ0 = 0.25 m, κ = 0.04) is last.

### 2. Radar features alone add almost nothing

| | Holdout | Val (context only) |
|---|---|---|
| R0, no radar | 0.4265 | 0.404 |
| R2, Doppler speed (best) | 0.4281 | 0.403 |
| The other 6 sets | 0.4179 to 0.4253 | 0.392 to 0.403 |

- **Holdout:** without the residual change, only R2 beats the radar-free network, and by 0.0016, which is within seed noise. R1 and R3–R7 all score below R0.
- **Val:** every radar set is at or below R0.
- **Richer sets do worse:** the sets with range residual and offset (R5, R6, R7) are lowest on val. These features describe where the returns lie inside the gate. They seem to fit train-split specifics rather than help.
- **Support:** adding the return count to Doppler (R3 against R2) costs 0.006 on holdout.

### 3. Only the residual change helps

The adoption threshold was 0.4298: R2's holdout mean plus the standard deviation of its seeds.

| Change on R2 | Holdout | Val | Verdict |
|---|---|---|---|
| **T2 residual on SECOND's score** | **0.4353** | **0.427** | adopted, final |
| T4b 10 motion-corrected sweeps | 0.4281 | 0.406 | no change |
| T4a 5 motion-corrected sweeps | 0.4261 | 0.404 | no change |
| T3 logit input | 0.4254 | 0.410 | no change |
| T5 RCS channel | 0.4225 | 0.395 | slightly worse |
| T1 far targets (IoU 0.5/0.3 beyond 50 m) | 0.4063 | 0.364 | clearly worse |

- **T2:** this was the only change adopted, so there was no combination to train.
- **T1:** the looser positive threshold for far candidates hurts both splits. Center-distance AP rewards well-placed boxes, and T1 teaches the network to keep boxes that are poorly placed.
- **T4a/T4b:** accumulating more radar sweeps, with Doppler motion correction, gives no measurable benefit over 3 sweeps.

### 4. The finals on val, 6 seeds

Seeds 4–6 were trained identically and added to seeds 1–3, to make the radar-gain estimate less noisy. The ranges below are the minimum and maximum over the 6 seeds (or 6 seed pairs).

| Condition | Cars beyond 50 m | SECOND | CLOCs R0+T2 | CLOCs-R R2+T2 | Radar gain | Pairs > 0 |
|---|---|---|---|---|---|---|
| All | 16,372 | **0.453** | 0.414 | 0.422 | +0.008 (−0.003 to +0.025) | 5 of 6 |
| Highway | 2,333 | **0.425** | 0.390 | 0.406 | +0.016 (−0.001 to +0.034) | 5 of 6 |
| Urban | 11,352 | **0.467** | 0.430 | 0.432 | +0.002 (−0.004 to +0.013) | 3 of 6 |
| Night | 1,349 | **0.445** | 0.425 | 0.434 | +0.009 (−0.008 to +0.032) | 3 of 6 |
| Rain | 1,338 | **0.479** | 0.421 | 0.473 | +0.052 (+0.002 to +0.115) | 6 of 6 |
| Adverse (night + rain) | 2,687 | **0.461** | 0.418 | 0.447 | +0.029 (−0.004 to +0.071) | 5 of 6 |

- **Rain:** this is the one condition where radar helps in every seed pair. The size of the gain varies a lot by seed, from +0.002 to +0.115.
- **Highway:** radar helps fairly consistently.
- **Urban and night:** no reliable gain. In urban scenes only half of the far cars have a radar return, and 40% of those are not moving, so their Doppler speed is near zero ([`../DATASET.md`](../DATASET.md)).
- **SECOND alone still wins beyond 50 m in every condition.** In rain, CLOCs-R comes within 0.006 of it.
- **Below 50 m fusion wins** (0.917 vs 0.901 at 0–30 m, 0.671 vs 0.645 at 30–50 m). There, radar makes no difference.
- **Recall at 100–200 m** (BEV IoU 0.3) is 0.614 for CLOCs and 0.649 for CLOCs-R, against 0.660 for SECOND. Radar mostly gives back cars that the camera-based fusion suppresses.
- **The 6-seed gain is smaller than the first round's 3-seed gain.** Seeds 4–6 had a smaller gain (+0.005 overall) than seeds 1–3 (+0.011), so the 6-seed figure (+0.008) is the better estimate. The first round reported +0.011 overall and +0.071 in rain.

### Camera removed at test time

The finals were scored with every 2D candidate deleted, which leaves SECOND's boxes and, for CLOCs-R, radar. This is an input the network almost never saw during training.

| Model | Holdout, with camera | Holdout, no camera | Val, with camera | Val, no camera |
|---|---|---|---|---|
| CLOCs R0+T2 | 0.427 | 0.424 | 0.414 | 0.454 |
| CLOCs-R R2+T2 | 0.435 | 0.431 | 0.422 | **0.461** |

The two splits disagree:

- **Holdout:** removing the camera costs a little, 0.004 overall and 0.036 at 100–200 m.
- **Val:** removing the camera *raises* far AP by about 0.04. CLOCs-R without the camera scores 0.461, above SECOND alone (0.453).

Val is more far-heavy than holdout, with 38% of its cars at 100–200 m against 12% in train. So on val the cost of the camera's missing far cars outweighs what the camera adds nearer in. This is an observation, not a configuration choice: no camera-free model was selected or trained. It does point to the next thing to try, which is giving the fusion network a way to tell "the camera cannot see this far" apart from "the camera sees nothing here". For example, it could use the 2D detector's expected recall at that range as an input.

### Latency

The radar features run on one CPU thread, in Python, on the median of 519 val frames:

| Set | Radar features per frame |
|---|---|
| R1 (support) | 3 ms |
| R2 (Doppler) | 19 ms |
| R7 (all five features) | 64 ms |

The fusion forward pass takes under 10 ms for every set on the RTX 4090.

## Caveats

- **Holdout cannot see far rain or far night.** It has no rain cars beyond 50 m and only 13 night cars at 100–200 m. So the selection could not reward radar where it helps most, and the rain and night results above are out-of-sample.
- **Val had already been read for the finals.** The first round read val for R0+T2 and R2+T2, and for their seeds 4–6. The ablation re-derived the same choice from holdout alone, so that earlier look did not steer it.
- **The ranges cover the fusion layer only.** They come from 6 fusion seeds, while SECOND and Cascade R-CNN were trained once. Night and rain have 2 val scenes each.

## Reproduce

```bash
source env.sh
python -m clocs_r.ablation_full                 # ~6 h on one RTX 4090; resumable
python results/full_dataset/ablation/make_tables.py
```

Weights: `runs/ablation/gate_*/R3_s{1,2,3}.pth` (step 1), `runs/ablation/sets_d0.5_k0.02/{config}_s{seed}.pth` (steps 2–4). Logs: [`../logs/ablation_full.txt`](../logs/ablation_full.txt). The weights are derived from aiMotive data and fall under its CC BY-NC-SA 4.0 license.
