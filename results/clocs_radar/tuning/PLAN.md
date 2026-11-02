# Fusion tuning plan

Written on 2026-11-01, after the ablation and before any tuning run. Val has already been seen for the as-designed models, so this tuning is disclosed as post hoc and reported alongside them, never in place of them.

**Starting point:** R2, the set the paper's rule selected on holdout. Detectors, gate and network size stay as they are.

**Metric:** center-distance AP beyond 50 m on holdout, mean of seeds 1, 2 and 3. Val is not read while choosing.

| Change | What it does | Problem it targets |
|---|---|---|
| T1 far targets | Beyond 50 m, a candidate is positive at BEV IoU of at least 0.5 and negative at 0.3 or less (0.7 and 0.5 elsewhere) | Few far candidates reach 0.7, so only 5% of positives are at 100-200 m |
| T2 residual | Fused logit = SECOND's logit + network output, with the last layer starting at zero | Fusion ranks far boxes worse than SECOND |
| T3 logit input | The network receives SECOND's score as logit / 5 instead of a probability | Differences between low scores are squeezed together |
| T4a 5 sweeps | M = 5, moving each return along its line of sight by its Doppler speed times its age | Only about 60% of far cars have a return |
| T4b 10 sweeps | The same with M = 10 | The same |
| T5 RCS | Adds the median radar cross-section of the gated returns, on a log scale, as a channel | Radar evidence does not separate cars from clutter well |

**Decision rule:**
1. Each change is tested alone against R2.
2. A change is adopted if its mean beats R2's by more than the standard deviation of R2's three seeds. R2 has a mean of 0.4554 and a standard deviation of 0.0028 on holdout, so a change must reach 0.4582. SECOND scores 0.4601.
3. If two or more are adopted, their combination is also tested. T4a and T4b are alternatives, so only the better of them can enter.
4. The highest mean among R2, the adopted changes and the combination wins. If none is adopted, R2 stands, and no tuned model goes to val.
5. The winner is scored once on val. The winning fusion changes are also trained without radar (R0), so the gain from radar can be told apart from the gain from repairing fusion.
