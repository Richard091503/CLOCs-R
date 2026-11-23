# Version 2: sparse LiDAR + radar fusion for far cars (plan draft, 2026-11-20)

A separate version of the project, on its own branch. It is not pre-registered yet: before anything is run, this
draft becomes the branch's `PLAN.md`, with dated rules, as the CLOCs version's `PLAN.md` is. Background:
`DECISIONS.md` items 9–10.

## Question

**Can sparse LiDAR points and sparse radar returns, fused at the point or feature level, detect cars beyond 50 m that
a LiDAR detector misses, with fewer false alarms than either sensor alone?**

The headline is the radar gain beyond 50 m: an identical recipe trained with and without radar, seed against seed. The
camera is left out at first, so it cannot confound the answer.

## Why

- On val, about 20% of cars beyond 50 m have only 1–5 LiDAR points. SECOND finds only 24–33% of them, and about half
  have a radar return. Another 11% have no LiDAR point at all.
- In the CLOCs version, LiDAR and radar meet only after SECOND has proposed a box, so a car SECOND never proposes cannot
  be recovered.
- A few LiDAR points plus a radar return with a car-like Doppler speed and RCS is much stronger evidence than either
  alone. Requiring both to agree should cut the false alarms that sank radar-only proposals in Phase 6 (95–99% false).

## Variants (each a single change against its no-radar control)

- **v2a, early fusion.** Radar returns join SECOND's point cloud as extra points, with a sensor flag, Doppler, RCS and
  sweep age. It is the same network and the same recipe; the control is SECOND trained identically without radar.
- **v2b, mid-level fusion.** Separate LiDAR and radar encoders, with their bird's-eye-view features merged by a
  distance-aware gate before one detection head.
  - Bi-LRFusion (CVPR 2023), LiRaFusion (ICRA 2024) and RadarNet (ECCV 2020) are the closest published designs. Check
    their details before relying on them.
  - RadarDistill's radar encoder (now built and trained in Phase 7) could serve as the radar branch.
- **v2c, a sensitive far-range stage.**
  - Proposals from small LiDAR clusters (even 1–2 points) and radar clusters together.
  - A small classifier confirms each proposal from both sensors: LiDAR point count, height and intensity, radar
    count, Doppler, RCS, the Phase 6b clutter probability, and persistence across frames.
  - Box size from a car prior, and heading from the lane direction or Doppler.
  - This is the most direct form of the idea, and the one that can be made deliberately sensitive.
- **v2d, densifying far LiDAR over time.** Accumulate a few past LiDAR sweeps, using radar Doppler to motion-correct
  moving far cars. It can be added to any of v2a–v2c.

## Cars without a ground-truth box

1. **Audit first.**
   - Project a sensitive detector's "false positives" beyond 100 m into the camera image, with the LiDAR and radar
     points, and count the real but unlabeled cars.
   - Measure how label density falls with range, using the `Occluded` and `Truncated` fields.
2. **Fill gaps along tracks.** Labels carry `ObjectId` and velocity, so a car labeled in some frames but missing in
   others can be carried along its track with the ego poses. These auto-labels are marked as such.
3. **Use future frames for training labels.** A car labeled when near can be traced back to frames where it was far.
   - This gives training labels annotators never drew.
   - The label-free counterpart is self-supervision from future frames, as in LRS4Fusion (ICCV 2025).
4. **Evaluation.**
   - The headline metric keeps the official labels.
   - A sensitivity analysis treats audited or track-filled cars as "ignore" regions, as KITTI's DontCare does, rather
     than false positives.
   - Never train and evaluate on the same auto-labels.

## Protocol (as in the CLOCs version, plus what was learned there)

- **Fixed splits:** `fit` for training, `select` for every choice.
  - Val is scored once per version, at the end, and its earlier exposure is disclosed.
  - The `select`/val mismatch on sparse far cars is a known limitation.
  - Re-drawing `select` would need retraining the detectors, and would have to be decided without val numbers.
- **Detector training costs hours per run,** so detector-level variants get 3 seeds, timed by a smoke test first. A
  pre-registered budget rule caps the schedule, as Phase 7's 24-hour rule does.
- **Statistics:** seed-paired differences and scene-bootstrap 95% intervals. Every variant tried is listed.
- **One heavy job at a time;** disk is checked before large writes.

## A bridge back to the CLOCs version (only if Phase 7 succeeds)

If Phase 7's radar detector and selector help, a v2c-style sparse-points stage could be added to the CLOCs pipeline,
for cars both detectors overlooked. Its outputs would be one more expert for the selector, not a source of unchecked
boxes. That would be its own pre-registered phase, after the CLOCs version's val scoring.

## First steps on the branch

1. Turn this draft into `PLAN.md`, with dated rules.
2. Label audit and range-coverage check (cheap).
3. Run v2a, the cheapest clean test. Its control is SECOND retrained identically, which also needs one more SECOND
   training run.
