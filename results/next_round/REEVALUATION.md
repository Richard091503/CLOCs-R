# Re-evaluation of the next round (2026-11-20, after Phase 6b's first audit, before Phase 7)

Written at the user's request, to check that the work would hold up to a reviewer, and to fix the structure for what
follows. Nothing here changes a past result. It records what each result can and cannot support, and what changes
from now on. The decisions are pre-registered in `PLAN.md`, Phase 7.

## 1. Data and the held-out split

- **aiMotive has no test split with labels:** only `train` and `val`.
- **Val is not untouched.**
  - It was scored once per phase in Phases 0–6, by each phase's pre-registered rule. That is 7 phases, plus the
    earlier `results/full_dataset` round.
  - No decision was made on val numbers, except the user's choice to run Phase 6 after seeing Phase 5's val ceiling.
    That choice is recorded in PLAN.md.
  - Even so, a reader should treat val as a **development set that has been looked at repeatedly**, not a clean test.
- **What we do about it:**
  1. Every val exposure is listed (section 5).
  2. All selection is done on `select`, which is never used for training.
  3. Phase 7's models are frozen before val is scored, and val is scored **once**, for all of them together.
  4. Claims are stated as effects on `select` (the selection data) and on val (reported with the exposure caveat),
     with uncertainty from both seeds and scenes (section 4).
- **`select` is easier than val for sparse far cars.** SECOND finds 55–62% of cars with 1–5 LiDAR points beyond 50 m on
  `select`, but 24–33% on val.
  - The split was fixed in Phase 1, before any of this was known.
  - Re-drawing it now, after seeing val, would be a decision informed by val. So it stays as it is, and the mismatch is
    reported as a limitation.

## 2. The framework

**CLOCs stays the backbone.** Detectors propose; a small fusion network rescores SECOND's candidates using the
agreement of other sensors. Radar must enter it in a way a reviewer can follow:

- **Radar enters any one model by exactly one route.** Either as per-candidate point features (CLOCs-R, R2), or as a
  radar detector's boxes paired with SECOND's candidates the way CLOCs pairs camera boxes. Never both.
  - Phase 6 used both at once, and also let radar boxes create new candidates. That made its result hard to interpret.
- **Only SECOND's candidates are rescored. Radar never creates boxes.**
  - This is CLOCs' own design.
  - Phases 6 and 6b measured radar-created candidates directly. On top of R2+T2+K40+G, they cost −0.104 APCD at
    50–200 m on `select`, and no filter of 25 gave a gain on `select`.
  - That line is closed, and reported as a negative result.
- **The camera is a confound for the radar question.**
  - Every model so far used the camera, and camera-LiDAR fusion loses 0.047 APCD to SECOND beyond 50 m on val.
  - To ask whether radar helps LiDAR beyond 50 m, Phase 7 adds **camera-free** CLOCs models. These are LiDAR + radar,
    with a LiDAR-only rescoring control. They are then combined with the camera model by the situational selector,
    which can also fall back to LiDAR alone.

## 3. What each earlier addition is, and how it is treated

| Addition | Chosen how | Status from Phase 7 on |
|---|---|---|
| SECOND, Cascade R-CNN on `fit` | Phase 1 rule (in-sample arm) | fixed |
| Residual scoring T2, far weight 2 | carried over from the previous round | fixed; ablated in `contributions.md` |
| K40 (top-40 2D boxes) | Phase 3 rule on `select` | fixed for camera pairs |
| Gated model G, condition flag C | Phase 4 rule (G adopted, C failed) | reported as a comparison; not used inside Phase 7's experts (the selector is Phase 7's only gate) |
| Range switch | Phase 2 baseline | reference line only; it cannot say anything about radar beyond 50 m |
| Radar-created candidates (RD), filters | Phases 6 and 6b | closed: negative result |
| Phase 5 oracle | analysis | reported as an upper bound only |

- These settings are **not revisited because of val**.
- The T2, far-weight and camera ablations in `contributions.md` are reported. They do not change Phase 7's settings,
  whatever they show.

## 4. Statistics

- **Seeds:** 6 per fusion model.
- **Primary comparisons:** paired seed against seed. Each reports the mean difference, the seed SD and the number of
  positive seeds.
- **Scene-level bootstrap:** 95% intervals from 1,000 resamples of scenes (all seeds averaged), so the uncertainty
  covers the data, not only training noise.
- **Every variant tried in Phases 0–7 is listed** in the final report, with its outcome, including the negative ones.

## 5. Val exposure log

| Round | What was scored on val |
|---|---|
| `results/full_dataset` (before next round) | the earlier models; the reason for the next round |
| Phase 0 | reproduction of the committed models |
| Phase 1 | R0+T2, R2+T2 (6 seeds), the detectors |
| Phase 2 | the range switch |
| Phase 3 | R2+T2+K40, R0+T2+K40 |
| Phase 4 | R2+T2+K40+G, R0+T2+K40+G |
| Phase 5 | the oracle ceiling, every model |
| Phase 6 | R2+T2+K40+G+RD, SECOND ∪ radar |
| Contributions (running) | R0, R2 without T2; R0+T2, R2+T2 at far weight 1; R2+T2+K40+G without the camera (reported, not selected on) |
| Phase 7 | once, at the end, every frozen Phase 7 model together |
