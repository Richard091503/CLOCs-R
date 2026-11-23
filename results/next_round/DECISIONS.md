# Design decisions and their reasons (next round, from 2026-11-17)

A dated record of the direction-setting discussions with the user. It complements `PLAN.md`, the pre-registered rules,
and `REEVALUATION.md`, the reviewer's-eye audit. Results live in each phase's report and in `README.md`.

## 2026-11-20

1. **The research question is whether radar helps beyond 50 m.**
   - The range switch (Phase 2) is SECOND alone beyond 50 m by construction, so it cannot answer this.
   - It stays in reports as a reference line, never as evidence about radar. The headline is the radar gain beyond
     50 m, radar against no radar with everything else fixed.
2. **CLOCs stays the backbone of this version.** Detectors propose, and a small fusion network rescores SECOND's
   candidates by the agreement of other sensors.
3. **Radar enters a model by one route only:** point features, or a radar detector's boxes. Never both in one model;
   Phase 6 did both, and its result was hard to interpret.
4. **Radar never creates candidates in this version.** Phases 6 and 6b showed that radar-only candidates were 95–99%
   false and cost up to −0.12 APCD beyond 50 m. That line is closed as a negative result.
5. **Radar through a detector is kept**, as CLOCs' second sensor in place of the camera (Phase 7, LRd). The candidates
   are the Phase 6 PointPillars (D1); D2, with motion-corrected sweeps and the clutter classifier's car probability;
   and RadarDistill (D3), using the authors' code.
6. **The situational selector is kept** (Phase 7, S). For each candidate it weights LiDAR alone, LiDAR + camera and
   LiDAR + radar, from each expert's score and the evidence behind it, with LiDAR alone as the fallback.
   - It is the user's original intent for "the switch": the system chooses by situation, rather than switching at a
     fixed range.
7. **Camera-free models are added**, because the camera confounds the radar question: camera-LiDAR fusion loses
   0.047 APCD to SECOND beyond 50 m on val.
8. **Research hygiene** (details in `REEVALUATION.md`):
   - Val has been looked at in every phase, so it is a development set. Phase 7 scores it once at the end.
   - Selection stays on `select`.
   - Settings are frozen and are not revisited because of val.
   - Seed-paired and scene-bootstrap statistics are reported, and every variant tried is listed.
9. **The sparse LiDAR + radar idea becomes its own version** (plan: `V2_PLAN.md`), developed on a separate branch.
   - In the CLOCs version, LiDAR and radar evidence only meet at the candidate level.
   - The new idea fuses sparse LiDAR and radar points directly, to detect far cars from a few points of each sensor and
     cut false alarms by requiring the two to agree.
10. **A possible bridge, only if Phase 7's detector and selector succeed:** a sparse-points "rescue" stage could be
    added to the CLOCs version for cars the detectors overlooked. This would be a later, separately pre-registered
    step, not part of Phase 7.
11. **Branches** (proposed; nothing committed until the user says so):
    - `master` stays at commit 688b38e.
    - The CLOCs version with Phases 0–7 goes on its own branch.
    - The sparse-points version gets its own branch, created as a git worktree so the running jobs are never disturbed.

## 2026-11-21

1. **The fixed-blend ablation B was added** (user's decision, before val; PLAN.md blend addendum).
   - It tests whether the selector's gain on `select` came from choosing by situation or from averaging experts.
   - B is the selector with its context removed: three trained weights, the same for every candidate.
   - It is post hoc and cannot become the current model.
2. **Finding: B matches or beats S** on `select` and on val.
   - On `select`, S − B = −0.0005 at 50–200 m (1/6 seeds favour S) and −0.0077 at 100–200 m (0/6 seeds favour S).
   - On val at 50–200 m, B 0.4428 against S 0.4407.
   - B's learned weights are the same for every seed: L 0.00, LC 0.72, LRd 0.28. The LiDAR-alone fallback is never
     used.
3. **Finding: the selector trusts the camera expert *more* far away**, by its mean weights on `select`:
   - at 0–30 m: LC 0.50–0.60, LRd 0.34–0.49;
   - at 100–200 m: LC 0.72–0.86, LRd 0.13–0.28.

   That is why the fixed blend beats it at 100–200 m. The likely causes, not yet measured:

   > **Correction (2026-11-21, 20:10 UTC).** The ranges above came from seed 1 only, the most extreme of the 6 seeds;
   > they were read from the first run in `selector.json`. Over all 6 seeds (candidate-weighted means over conditions,
   > `select`), the pattern is milder:
   >
   > | Expert | 0–30 m | 30–50 m | 100–200 m |
   > |---|---|---|---|
   > | LC | 0.58–0.64 | | 0.63–0.77 |
   > | LRd | 0.29–0.37 | 0.18–0.25 | 0.23–0.36 |
   >
   > - The camera weight rises by about 0.1 with distance. The radar weight dips at 30–100 m and comes back at 100–200 m.
   >   The weight that falls with distance is mostly L's (about 0.05–0.10 near, about 0 at 100–200 m).
   > - At 100–200 m, S's mean radar weight (about 0.30) is close to B's fixed 0.28. **S's loss to B at 100–200 m is
   >   therefore not explained by a camera lean on average.** It is better explained by S's weights varying from
   >   candidate to candidate, which reorders scores within the bin (third cause below).
   > - The causes below are kept as written, for the record. The first (no radar-detector evidence in S's context) is
   >   still a fact of the design, but it is no longer the leading explanation.
   > - The figure `figures/phase7_weights.png` and `phase7/report.md` section 6 show the 6-seed means.
   - **A design gap in S, found in review:** its context has camera evidence (the best matching 2D box's IoU × score)
     but no radar-detector evidence. It sees only raw radar return counts, never whether D2 found a box there. S can
     tell when the camera saw something, but not when the radar detector did.
   - **The loss and the metric disagree far away:** 112,662 candidates for 2,760 cars at 100–200 m on `select`. The
     focal loss rewards suppressing false candidates, which the camera expert does well, while far-range APCD
     depends on ranking the few true cars.
   - **Per-candidate weights reorder scores across candidates,** which a single fixed blend cannot do.
4. **Finding: the radar-feature gain (LRf) did not carry over to val**, while the radar-detector gain (LRd) did. On val
   at 50–200 m (6 seeds, before the bootstrap):

   | Model | Val APCD |
   |---|---|
   | L0 | 0.4282 |
   | LRf | 0.4233 |
   | LRd | 0.4415 |

   - LRd − L0 = +0.013, ahead on 6/6 seeds; on `select` it was +0.015, also 6/6.
   - LRf − L0 = −0.005, behind on 6/6 seeds, against +0.012 on `select`.
5. **Further selector work is gated by an oracle ceiling** (user approved; PLAN.md ceiling addendum).
   - Only if a perfect per-distance blend, tuned on `select`, beats B by at least 0.005 at 100–200 m do two variants
     run: S plus radar-detector evidence, and a selector over the three detectors' own scores.
   - The three-detector selector would use D2, with RadarDistill (D3) as a named comparison.
   - Reasons for not running the full six-variant round:
     - the main question is answered by LRd − L0;
     - B leaves little headroom;
     - every new variant now comes after val was seen, so many variants would mean many forking paths.
6. **The order after the ceiling check** (user approved):
   1. the variants, only if the rule allows;
   2. the Phase 7 report;
   3. cut the CLOCs version down to what earned its place;
   4. then the sparse LiDAR + radar version (`V2_PLAN.md`).
7. **Documentation for the thesis** (user, 2026-11-21): the project is the basis of the user's thesis. Every step,
   decision and finding, negative ones included, is recorded here, in `PLAN.md` (rules before runs), in the phase
   reports and `README.md` (results), and in `STATUS.md` (current state).

8. **Finding: the Phase 7 scene bootstraps** (`phase7/bootstrap_select.json`, `bootstrap_val.json`; 1,000 scene
   resamples, means over 6 seeds, 95% intervals; `select` has 24 scenes and val 25).
   - The point estimates equal the stored APCD (to within 2e-6).

   APCD differences at 50–200 m (100–200 m in the second column):

   | Comparison | `select` 50–200 | `select` 100–200 | val 50–200 | val 100–200 |
   |---|---|---|---|---|
   | LRd − L0 | +0.0148 [+0.0070, +0.0218] | +0.0100 [+0.0017, +0.0170] | +0.0133 [+0.0049, +0.0228] | +0.0129 [+0.0078, +0.0179] |
   | LRf − L0 | +0.0124 [−0.0018, +0.0262] | +0.0076 [−0.0065, +0.0208] | −0.0049 [−0.0128, +0.0039] | +0.0003 [−0.0065, +0.0060] |
   | LRd − LRf | +0.0024 [−0.0078, +0.0128] | +0.0024 [−0.0103, +0.0135] | +0.0182 [+0.0104, +0.0238] | +0.0126 [+0.0068, +0.0191] |
   | LRd − L | +0.0114 [+0.0032, +0.0185] | +0.0110 [+0.0032, +0.0174] | −0.0040 [−0.0146, +0.0050] | +0.0016 [−0.0042, +0.0079] |
   | S − LC | +0.0121 [+0.0073, +0.0172] | +0.0088 [+0.0020, +0.0156] | +0.0143 [+0.0084, +0.0196] | +0.0087 [+0.0008, +0.0181] |
   | S − B | −0.0005 [−0.0040, +0.0038] | −0.0077 [−0.0140, −0.0022] | −0.0022 [−0.0052, +0.0018] | −0.0035 [−0.0076, +0.0015] |
   | S − L | +0.0289 [+0.0084, +0.0516] | +0.0133 [−0.0081, +0.0358] | −0.0048 [−0.0139, +0.0038] | +0.0035 [−0.0075, +0.0156] |
   | B − L | +0.0294 [+0.0121, +0.0480] | +0.0210 [+0.0014, +0.0410] | −0.0027 [−0.0130, +0.0057] | +0.0070 [−0.0057, +0.0199] |
   | S − incumbent | +0.0072 [+0.0010, +0.0133] | +0.0164 [+0.0032, +0.0340] | +0.0207 [+0.0132, +0.0291] | +0.0264 [+0.0119, +0.0422] |
   | LC − L | +0.0168 [−0.0055, +0.0424] | +0.0045 [−0.0189, +0.0298] | −0.0191 [−0.0312, −0.0071] | −0.0052 [−0.0231, +0.0135] |

   Val APCD, 50–200 m (100–200 m in brackets):

   | Model | Val APCD |
   |---|---|
   | L (SECOND alone) | **0.4455** (0.2841) |
   | L0 | 0.4282 (0.2729) |
   | LC | 0.4264 (0.2789) |
   | LRf | 0.4233 (0.2731) |
   | LRd | 0.4415 (0.2858) |
   | S | 0.4407 (0.2877) |
   | B | 0.4428 (0.2912) |
   | Incumbent | 0.4199 (0.2613) |

   **What this says:**
   - **Radar through a detector helps within CLOCs on both splits.** LRd − L0 has intervals above zero in both far
     bins, on `select` and on val. Radar as point features does not: its `select` gain vanished on val.
   - **On val, no fusion model beats SECOND alone beyond 50 m.** SECOND alone has the highest val APCD at 50–200 m,
     and every fusion − L interval at 50–200 m includes zero or is below it.
     - The camera-free CLOCs control L0 is itself about 0.017 below SECOND on val. Radar recovers most of that, but
       not more.
     - On `select`, the fusion models were clearly above SECOND, so the fusion layer's advantage over SECOND does not
       carry over from `select` to val.
   - **A pattern worth checking:** on val, LRd is above L in each of 50–100 m and 100–200 m separately, but below it
     in the pooled 50–200 m bin. That points to the fusion's scores being ranked less consistently across distances
     than SECOND's. It is a hypothesis, not yet tested.
   - **The fixed blend is at least as good as the selector on both splits.**

9. **Finding: on val, the blend's loss to SECOND is only in the pooled bins.** Val APCD per bin (6-seed means):

   | Model | 0–30 m | 30–50 m | 50–100 m | 100–200 m | 50–200 m pooled | All pooled |
   |---|---|---|---|---|---|---|
   | L | 0.8971 | 0.6206 | 0.5992 | 0.2841 | 0.4455 | 0.5401 |
   | B | 0.9176 | 0.6398 | 0.5985 | 0.2912 | 0.4428 | 0.5376 |
   | LRd | 0.8913 | 0.6213 | 0.6028 | 0.2858 | 0.4415 | 0.5126 |

   - B is above or level with SECOND in every bin, yet below it in both pooled bins. Its fused scores rank well within
     each range but are not comparable across ranges.
   - Val's 50–200 m bin is 50% cars at 100–200 m, against 29% on `select`, so this costs more on val. That accounts for
     part of the disagreement between `select` and val.
   - **Considered and not pursued** (user's decision, 2026-11-21): a per-distance-bin calibration of the fused score,
     fitted on `fit`'s out-of-fold scores. It would leave every per-bin APCD unchanged and change only the pooled bins.
   - The thesis should report the per-bin breakdown alongside the pre-registered pooled metrics. The headline metric is
     not changed after the fact.

10. **Finding: the oracle ceiling of situation-dependent blends is small,** so selector work stops by the
    pre-registered rule (PLAN.md ceiling addendum; `phase7/ceiling.json`; `select`, 6 seeds, 231 weightings).
    - **Tool check:** it reproduces B's and S's stored APCD to within 3.6e-5; the tolerance was 5e-4.
    - **Each oracle minus B, on `select`:**

      | Upper bound | 0–30 m | 30–50 m | 50–100 m | 100–200 m | 50–200 m |
      |---|---|---|---|---|---|
      | Global (best single w per bin) | +0.0017 | +0.0003 | +0.0009 | +0.0023 | +0.0009 |
      | Distance (best w per bin) | +0.0010 | −0.0003 | +0.0008 | **+0.0025** | +0.0015 |
      | Distance × condition (annotation labels) | −0.0044 | +0.0014 | +0.0017 | +0.0043 | +0.0030 |

    - **Rule:** the distance oracle minus B is +0.0025 at 100–200 m, below the 0.005 required, so the two variants are
      not run. They were S plus radar-detector evidence, and the three-detector selector.
    - **The distance oracle minus the global oracle is about 0 in every bin** (−0.0007 to +0.0006). Letting the weights
      depend on distance adds nothing beyond tuning a single blend on `select`.
    - **Even the distance × condition oracle,** which uses annotation labels not available at run time, adds at most
      +0.0043.
    - **The distance oracle's weights** give LiDAR alone weight only at 0–30 m (0.0–0.25). Beyond 30 m it is LC 0.50–0.75
      and LRd 0.25–0.50, with more radar weight at 100–200 m (0.45–0.50) than B has (0.28). The gain from that shift is
      within +0.0025.
    - **Conclusion for the thesis:** on this data, a fixed blend of the camera and radar CLOCs experts captures nearly
      all that situation-dependent weighting of these experts could capture.

11. **The final CLOCs model is kept to one network** (user's decision, 2026-11-21).
    - The user rejected both the selector and the two-network blend as over-engineered for a final product, and asked
      to stay close to CLOCs' structure.
    - The design: **one CLOCs network whose pair cells come from all three detectors** (SECOND's candidates; Cascade
      R-CNN's 2D boxes; D2's radar boxes).
      - Each candidate is fused with the camera, the radar detector, both or neither, according to which detectors
        matched it.
      - No separate selector, and each sensor enters once.
    - Pre-registered in the PLAN.md LCR addendum. It is post-val development, and it becomes final only if it is not
      worse than B on `select` (by its own seed SD, both far bins).
    - D2 is kept over RadarDistill, as the pre-registered rule chose it (0.217 against 0.215 alone), so the comparison
      with LRd stays like-for-like.

12. **Finding: the one-network LCR fails its rule on `select`, so B stays the final model**
    (`phase7/decision_lcr.json`; 6 seeds, trained 2026-11-21 20:32–21:04 UTC).

    | Model | 0–30 m | 30–50 m | 50–100 m | 100–200 m | 50–200 m |
    |---|---|---|---|---|---|
    | L | 0.9072 | 0.7797 | 0.6277 | 0.4052 | 0.5639 |
    | LC | 0.9066 | 0.7864 | 0.6497 | 0.4096 | 0.5807 |
    | LRd | 0.9080 | 0.7759 | 0.6431 | 0.4162 | 0.5753 |
    | **LCR** | 0.9095 | 0.7812 | 0.6509 | 0.4125 ± 0.0043 | 0.5817 ± 0.0022 |
    | B | 0.9100 | 0.7921 | 0.6617 | 0.4261 | 0.5933 |

    Seed-paired differences on `select`:

    | Comparison | 50–200 m | 100–200 m |
    |---|---|---|
    | LCR − B | −0.0116 (0/6) | −0.0136 (0/6) |
    | LCR − LC | +0.0010 (4/6) | +0.0029 (4/6) |
    | LCR − LRd | +0.0064 (6/6) | −0.0037 (2/6) |

    - **Adding the radar-detector cells to the camera network barely changes it.** LCR ≈ LC. Two separate networks,
      blended, are clearly better.
    - **Hypothesis, not tested:** Cascade R-CNN and D2 were both trained on `fit`, the fusion network's own training
      split. On `fit` the camera evidence looks far more reliable than on unseen data (Phase 1: Cascade R-CNN finds 92%
      of 100–200 m cars on its training scenes, against 23% on val). A single network learns to lean on the camera and
      under-uses radar.
      - The separate radar network has no camera to lean on, so it learns radar.
      - This matches the Phase 1 cross-fitting observation: with cross-fitted inputs, the radar gain on `select` was
        +0.009 (3/3), against +0.000 in-sample.
      - Cross-fitted detector outputs would be the direct test. They have not been run.
      - *Precision added 2026-11-22:* the 92% is the Phase 0 Cascade R-CNN on `train`, its own training scenes
        (README, Phase 1 table). The in-sample recall of the `fit`-trained Cascade R-CNN on `fit` was not measured.
        It has 35.3% on `select` and 22.7% on val.
    - As pre-registered, val and the bootstrap are still scored for LCR.

13. **LCR is the main model of the CLOCs version, by the user's choice, overriding the pre-registered rule**
    (2026-11-21, about 22:05 UTC).
    - **Reason given:** LCR is simpler (one CLOCs network over the three detectors, against two networks plus a blend)
      and performs well in its own right.
    - **What was known when the choice was made:** the `select` results in item 12. LCR's val score existed (21:19 UTC),
      but the user had not been shown it yet.
    - **The rule said B** (item 12). The override is recorded as a departure from the pre-registration, made for
      simplicity, not on the numbers.
    - **B, LC and LRd stay in every table** as comparisons. No code or result is removed.
    - **LCR on val** (scored once, frozen weights, 6 seeds, APCD mean ± SD):

      | Model | all | 0–30 | 30–50 | 50–100 | 100–200 | 50–200 |
      |---|---|---|---|---|---|---|
      | L | 0.5401 | 0.8971 | 0.6206 | 0.5992 | 0.2841 | 0.4455 |
      | LC | 0.5247 | 0.9158 | 0.6304 | 0.5767 | 0.2789 | 0.4264 |
      | LRd | 0.5126 | 0.8913 | 0.6213 | 0.6028 | 0.2858 | 0.4415 |
      | B | 0.5376 | 0.9176 | 0.6398 | 0.5985 | 0.2912 | 0.4428 |
      | **LCR** | **0.5254 ± 0.0014** | **0.9132 ± 0.0037** | **0.6294 ± 0.0028** | **0.5803 ± 0.0039** | **0.2779 ± 0.0071** | **0.4271 ± 0.0017** |

    - **On val, LCR behaves like camera CLOCs (LC):**
      - It beats SECOND alone within 50 m: +0.016 at 0–30 m and +0.009 at 30–50 m.
      - It is below SECOND beyond 50 m: −0.019 at 50–100 m, −0.006 at 100–200 m and −0.018 at 50–200 m.
      - The radar cells add almost nothing over LC: +0.0007 at 50–200 m and −0.001 at 100–200 m.
    - **Consequence for the research question:** with LCR as the main model, the CLOCs version does not show a radar
      gain beyond 50 m on val. The radar gain is in LRd (LRd − L0 +0.013 at 50–200 m, bootstrap interval excludes 0;
      item 8) and is not picked up by the single network. The hypothesis in item 12 (in-sample camera evidence) is the
      leading explanation and remains untested.
    - **Scene bootstrap** (finished 2026-11-22 00:18 UTC; 1,000 scene resamples, means over 6 seeds, 95% intervals;
      `phase7/bootstrap_lcr_{select,val}.json`):

      | Comparison | split | 50–100 | 100–200 | 50–200 | all |
      |---|---|---|---|---|---|
      | LCR − LC | select | +0.0012 [−0.0028, +0.0056] | +0.0029 [+0.0001, +0.0067] | +0.0010 [−0.0022, +0.0047] | +0.0002 [−0.0036, +0.0034] |
      | LCR − LC | val | +0.0036 [−0.0030, +0.0111] | −0.0010 [−0.0053, +0.0029] | +0.0007 [−0.0042, +0.0051] | +0.0008 [−0.0025, +0.0039] |
      | LCR − LRd | select | +0.0078 [−0.0128, +0.0326] | −0.0037 [−0.0240, +0.0210] | +0.0064 [−0.0117, +0.0278] | +0.0080 [+0.0002, +0.0178] |
      | LCR − LRd | val | −0.0225 [−0.0331, −0.0100] | −0.0078 [−0.0240, +0.0092] | −0.0144 [−0.0221, −0.0041] | +0.0129 [+0.0048, +0.0218] |
      | LCR − B | select | −0.0108 [−0.0169, −0.0029] | −0.0136 [−0.0213, −0.0060] | −0.0116 [−0.0174, −0.0041] | −0.0090 [−0.0126, −0.0057] |
      | LCR − B | val | −0.0182 [−0.0260, −0.0085] | −0.0132 [−0.0223, −0.0046] | −0.0157 [−0.0220, −0.0091] | −0.0122 [−0.0175, −0.0066] |
      | LCR − L | select | +0.0232 [−0.0020, +0.0515] | +0.0074 [−0.0156, +0.0323] | +0.0178 [−0.0036, +0.0429] | +0.0060 [−0.0036, +0.0183] |
      | LCR − L | val | −0.0189 [−0.0362, −0.0030] | −0.0062 [−0.0271, +0.0145] | −0.0184 [−0.0332, −0.0046] | −0.0146 [−0.0270, −0.0023] |

      - **LCR is reliably below B** on both splits, in every far bin and overall.
      - **The radar cells' contribution (LCR − LC) is not distinguishable from 0** on either split, except a borderline
        +0.003 at 100–200 m on `select`.
      - **On val, LCR is reliably below SECOND alone** at 50–100 m, 50–200 m and overall, and below LRd at 50–100 m and
        50–200 m.
      - **On `select`, LCR's gain over SECOND** (+0.018 at 50–200 m) has an interval that includes 0.

14. **LCR-D is tested: camera dropout beyond 50 m in training** (user's choice, 2026-11-21, pre-registered 22:25 UTC
    in the PLAN.md LCR-D addendum).
    - **The user's question:** can the single network be made to trust radar more beyond 50 m?
    - **Options discussed and not run:**
      - **A rule-based or learned switch before the network**, choosing which detectors' outputs it sees. Not run: the
        oracle ceiling (item 10) bounds any switch among L, LC and LRd by distance at +0.0025, and by distance and
        condition at +0.004, over B on `select`. It also reintroduces the selector machinery the cut-down removes.
      - **A hard switch with no camera beyond 50 m (LCR-H).** Not run: it hard-codes a choice suggested by val.
      - **Inflating the radar detector's scores.** Not run:
        - inflating in training and testing is undone by the network's first layer;
        - inflating at test time only is a factor to tune, so it would be tuned on `select` (likely to about 1) or on
          val, which is ruled out;
        - it also boosts clutter matches.
    - **Chosen: LCR-D,** which takes the camera crutch away in training so the network learns radar, and leaves test
      time unchanged. It is the closest to "trust radar more when the radar evidence is there", with no knob tuned.
    - **Implementation:**
      - `fusion.drop_far_camera`;
      - `train_fusion.fit(far_cam_dropout=...)`;
      - token DF in `tune_fusion` (requires RC; `DF_M` = 50 m, `DF_P` = 0.5).
      - Tests in `test_core.py`: only the camera cells of far candidates are dropped; radar cells stay; a candidate left
        with nothing gets the reserved cell; the drop rate is about 0.5; LCR's settings are unchanged; the cached cells
        are not modified.
    - **Run:** `results/next_round/phase7_lcrd.sh` (log `runs/next_round/phase7/lcrd.log`). It is queued behind the LCR
      bootstrap, so only one heavy job runs at a time.
    - **Result on `select`** (2026-11-22 00:51 UTC; 6 seeds; difference paired by seed, with the seeds where LCR-D is
      higher):

      | Bin | LCR-D | LCR | LCR-D − LCR |
      |---|---|---|---|
      | all | 0.7250 ± 0.0010 | 0.7252 | −0.0002 (3/6) |
      | 0–30 | 0.9107 ± 0.0007 | 0.9095 | +0.0012 (6/6) |
      | 30–50 | 0.7790 ± 0.0012 | 0.7812 | −0.0022 (1/6) |
      | 50–100 | 0.6468 ± 0.0024 | 0.6509 | −0.0041 (1/6) |
      | 100–200 | 0.4182 ± 0.0022 | 0.4125 | +0.0056 (5/6) |
      | 50–200 | 0.5797 ± 0.0021 | 0.5817 | −0.0020 (2/6) |

      - **The rule keeps LCR** (`phase7/decision_lcrd.json`). LCR-D fails at 30–50 m: −0.0022, more than its SD of
        0.0012. It passes at 0–30, 50–200 and 100–200 m.
      - **At 100–200 m, the dropout does make the network use radar:**
        - LCR-D 0.4182, against LCR 0.4125, LC 0.4096 and LRd 0.4162, so LCR-D − LC = +0.0086;
        - B is still higher (0.4261).
      - **The cost is at 30–100 m**, where the camera is still informative and the network now trusts it less.
      - Val and the bootstrap are still scored, as pre-registered.
    - **Result on val** (scored once from frozen weights, 01:06 UTC):

      | Bin | LCR-D | LCR | LCR-D − LCR | L (SECOND) |
      |---|---|---|---|---|
      | all | 0.5315 ± 0.0041 | 0.5254 | +0.0061 (6/6) | 0.5401 |
      | 0–30 | 0.9087 ± 0.0012 | 0.9132 | −0.0045 (1/6) | 0.8971 |
      | 30–50 | 0.6274 ± 0.0020 | 0.6294 | −0.0021 (1/6) | 0.6206 |
      | 50–100 | 0.5901 ± 0.0042 | 0.5803 | +0.0097 (6/6) | 0.5992 |
      | 100–200 | 0.2801 ± 0.0064 | 0.2779 | +0.0022 (2/6) | 0.2841 |
      | 50–200 | 0.4347 ± 0.0056 | 0.4271 | +0.0076 (6/6) | 0.4455 |

      - **On val, the dropout recovers part of LCR's far-range loss:** +0.0076 at 50–200 m (6/6), mostly at 50–100 m.
      - **LCR-D is still below SECOND alone beyond 50 m:** −0.009 at 50–100 m, −0.004 at 100–200 m and −0.011 at
        50–200 m. Within 50 m it stays above SECOND: +0.012 at 0–30 m and +0.007 at 30–50 m.
      - **Where the gain lands differs between splits:** on `select` it is at 100–200 m, with a loss at 50–100 m; on val
        it is at 50–100 m. Val's seed SD (0.004–0.006) is two to three times `select`'s.
      - By the pre-registered rule LCR stays. This val result is val-informed, so it does not change that.
    - **Scene bootstrap** (finished 04:01 UTC; 1,000 scene resamples, means over 6 seeds, 95% intervals;
      `phase7/bootstrap_lcrd_{select,val}.json`):

      | Comparison | split | 50–100 | 100–200 | 50–200 | all |
      |---|---|---|---|---|---|
      | LCR-D − LCR | select | −0.0041 [−0.0183, +0.0085] | +0.0056 [−0.0091, +0.0193] | −0.0020 [−0.0145, +0.0096] | −0.0002 [−0.0060, +0.0052] |
      | LCR-D − LCR | val | +0.0097 [−0.0029, +0.0242] | +0.0022 [−0.0149, +0.0204] | +0.0076 [−0.0030, +0.0184] | +0.0061 [−0.0016, +0.0149] |
      | LCR-D − LC | select | −0.0029 [−0.0188, +0.0120] | +0.0086 [−0.0063, +0.0225] | −0.0010 [−0.0143, +0.0124] | +0.0000 [−0.0082, +0.0073] |
      | LCR-D − LC | val | +0.0133 [−0.0028, +0.0325] | +0.0012 [−0.0148, +0.0171] | +0.0083 [−0.0035, +0.0200] | +0.0069 [−0.0028, +0.0160] |
      | LCR-D − LRd | select | +0.0037 [−0.0091, +0.0175] | +0.0020 [−0.0109, +0.0181] | +0.0044 [−0.0071, +0.0176] | +0.0078 [+0.0030, +0.0135] |
      | LCR-D − LRd | val | −0.0127 [−0.0229, −0.0023] | −0.0057 [−0.0119, −0.0003] | −0.0068 [−0.0139, +0.0008] | +0.0190 [+0.0115, +0.0289] |
      | LCR-D − L | select | +0.0191 [+0.0008, +0.0377] | +0.0130 [−0.0048, +0.0322] | +0.0158 [−0.0006, +0.0318] | +0.0058 [−0.0034, +0.0150] |
      | LCR-D − L | val | −0.0091 [−0.0284, +0.0123] | −0.0040 [−0.0136, +0.0048] | −0.0108 [−0.0225, +0.0019] | −0.0085 [−0.0191, +0.0038] |

      - **Every LCR-D − LCR interval includes 0, on both splits.** Neither the `select` loss at 30–100 m nor the val gain
        beyond 50 m is distinguishable from scene-to-scene variation. The intervals are wider than LCR's (for example
        LCR − LC at 50–200 m on `select` spans 0.007, LCR-D − LC spans 0.027), so the dropout makes the network's far
        scores depend more on the scene.
      - **Against SECOND alone, LCR-D is no longer reliably worse on val** at 50–200 m: −0.0108 [−0.0225, +0.0019],
        against LCR's −0.0184 [−0.0332, −0.0046]. It is still below on the point estimate.
      - **On val, LCR-D is reliably below the radar network LRd** at 50–100 m and 100–200 m. Overall it is reliably above
        LRd on both splits.
      - **Conclusion:** the camera dropout moves the single network towards radar without a reliable gain. LCR stays
        the main model by the rule. LCR-D is reported as a tested variant.

15. **The project is split into two branches** (user's decision, 2026-11-22).
    - **The user's reasons:**
      - LCR is the model for the autonomous-vehicles class paper; the rest of the work goes to the thesis.
      - Clean up the codebase and save space, while keeping every piece of information gathered.
      - Not every model needs to stay executable, as long as its structure is documented well enough to rebuild it.
    - **`thesis` (this branch):** the whole record, unchanged and executable. Every model, script, report and pre-registration.
    - **`master`: the LCR version.**
      - Code: the LCR pipeline only, with `clocs_r/run_lcr.sh` from data to bootstrap. L, LC, LRd, LCR and LCR-D are
        executable.
      - `tune_fusion` keeps the tokens R0, T2, K40, RP, RC and DF. Its settings dicts are identical to this branch's, so
        the saved networks load; `test_core` checks this against the saved weights.
      - `bootstrap.py` no longer needs `selector.py`.
      - `docs/MODELS.md` describes every other model.
      - `results/lcr/` holds `REPORT.md`, generated from raw result files copied unchanged from `phase*/`, and the 24
        trained fusion networks.
      - Removed from `master` (still here): the earlier CLOCs-R code and results, the range switch, gating, injection,
        RD, the radar filters, D1, D3, the selector, the blend, the ceiling, the cross-fitting scripts, and the KITTI
        reproduction with the original CLOCs `second/` and `torchplus/` trees.
      - The `master` checkout went from about 75 MB to about 3 MB (`.git` and `runs/` not counted).
    - **Run outputs** stay in the `master` checkout's `runs/` (not in git). `runs/lcr/` is made of links into
      `runs/next_round/`, so nothing was moved or copied.
    - **Not deleted:** any run output. Disk savings on the VM are left to the user (see `STATUS.md`).
    - **Check (2026-11-22, 05:10 UTC):** `master`'s code re-scored the saved networks of LC, LRd, LCR and LCR-D on
      `select` (seeds 1–3). It matches the stored results in every bin and condition, with a largest difference of 0.
      Its bootstrap point estimates match the stored seed-1 APCD to within 1.7e-6. `test_core` passes on both branches.

## Data facts behind these decisions (measured 2026-11-20)

**Scored cars beyond 50 m, by LiDAR points in the box** (val; `select` in brackets):

| LiDAR points | Share of far cars | With a radar return | Found by SECOND |
|---|---|---|---|
| 0 | 10.9% | 48% | 24% (27%) |
| 1–2 | 7.6% | 52% | 24% (55%) |
| 3–5 | 12.3% | 51% | 33% (62%) |
| 6–20 | 36.1% | 67% | 69% (88%) |
| 21+ | 33.1% | 67% | 93% (97%) |

- About 20% of far cars on val have 1–5 LiDAR points. SECOND finds only a quarter to a third of them, and about half
  have a radar return. That is the opportunity for the sparse-points version.
- **Labels carry** track IDs (`ObjectId`), absolute and relative velocity, and `Occluded` and `Truncated` flags.
- **Radar is sparse:** about 106 returns per frame with 5 sweeps, and only about 21% of car boxes have a radar return
  in the gate.
- **The per-point clutter classifier** (Phase 6b) separates car returns from clutter with AUC 0.93 on `select` (0.90
  beyond 50 m).
