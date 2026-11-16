## 1. Gate (R3, holdout)

Center-distance AP beyond 50 m, mean of seeds 1-3. SECOND alone: 0.4395.

| Gate | Mean | Seeds |  |
|---|---|---|---|
| **δ0 = 0.5 m, κ = 0.02** | 0.4222 | 0.4160, 0.4237, 0.4269 | selected |
| δ0 = 0.25 m, κ = 0.01 | 0.4220 | 0.4254, 0.4167, 0.4239 |  |
| δ0 = 0.5 m, κ = 0.04 | 0.4216 | 0.4163, 0.4213, 0.4272 |  |
| δ0 = 1 m, κ = 0.01 | 0.4214 | 0.4159, 0.4206, 0.4277 |  |
| δ0 = 0.5 m, κ = 0.01 | 0.4214 | 0.4219, 0.4197, 0.4225 |  |
| δ0 = 1 m, κ = 0.04 | 0.4210 | 0.4199, 0.4173, 0.4259 |  |
| δ0 = 1 m, κ = 0.02 | 0.4208 | 0.4150, 0.4194, 0.4280 |  |
| δ0 = 0.25 m, κ = 0.02 | 0.4201 | 0.4115, 0.4194, 0.4295 |  |
| δ0 = 0.25 m, κ = 0.04 | 0.4178 | 0.4124, 0.4151, 0.4259 |  |


## 2. Feature sets (gate δ0 = 0.5 m, κ = 0.02)

Holdout chose; val is shown for context and played no part. SECOND alone: holdout 0.4395, val 0.4533. r = number of radar returns in the gate (log), v = Doppler speed, e = Doppler spread, Δ = range residual, o = offset.

| Set | Features | Holdout mean | Holdout seeds | Val (seeds 1-3) |  |
|---|---|---|---|---|---|
| R0 | none | 0.4265 | 0.4257, 0.4297, 0.4242 | 0.404 (0.398 to 0.407) | reference |
| R1 | r | 0.4225 | 0.4257, 0.4190, 0.4230 | 0.403 (0.398 to 0.408) |  |
| **R2** | v | 0.4281 | 0.4261, 0.4288, 0.4293 | 0.403 (0.400 to 0.406) | selected |
| R3 | r, v | 0.4222 | 0.4160, 0.4237, 0.4269 | 0.399 (0.395 to 0.402) |  |
| R4 | r, v, e | 0.4229 | 0.4142, 0.4298, 0.4247 | 0.399 (0.392 to 0.406) |  |
| R5 | r, Δ, o | 0.4179 | 0.4134, 0.4212, 0.4191 | 0.392 (0.389 to 0.397) |  |
| R6 | r, v, Δ | 0.4197 | 0.4224, 0.4128, 0.4240 | 0.393 (0.381 to 0.401) |  |
| R7 | r, v, e, Δ, o | 0.4253 | 0.4223, 0.4311, 0.4225 | 0.393 (0.392 to 0.396) |  |


## 3. Fusion changes on R2

A change is adopted if its holdout mean exceeds R2's by more than the standard deviation of R2's seeds, that is above **0.4298**. Adopted: T2.

| Configuration | Change | Holdout mean | Holdout seeds | Val (seeds 1-3) | Role |
|---|---|---|---|---|---|
| R2 | none (base) | 0.4281 |  | 0.403 (0.400 to 0.406) | base |
| **R2+T2** | residual on SECOND's score | 0.4353 | 0.4350, 0.4344, 0.4365 | 0.427 (0.422 to 0.430) | adopted, **final** |
| R2+T1 | far targets | 0.4063 | 0.4050, 0.4092, 0.4045 | 0.364 (0.360 to 0.368) | not adopted |
| R2+T3 | logit input | 0.4254 | 0.4239, 0.4228, 0.4295 | 0.410 (0.403 to 0.415) | not adopted |
| R2+T4a | 5 sweeps, motion-corrected | 0.4261 | 0.4192, 0.4324, 0.4266 | 0.404 (0.399 to 0.409) | not adopted |
| R2+T4b | 10 sweeps, motion-corrected | 0.4281 | 0.4261, 0.4297, 0.4285 | 0.406 (0.401 to 0.414) | not adopted |
| R2+T5 | RCS channel | 0.4225 | 0.4240, 0.4190, 0.4244 | 0.395 (0.392 to 0.397) | not adopted |


## 4. Final configurations on val, cars beyond 50 m (center-distance AP, 6 seeds)

Final CLOCs-R: **R2+T2**. Final CLOCs: **R0+T2**. Radar gain is computed seed against seed.

| Condition | Cars | SECOND | CLOCs (R0+T2) | CLOCs-R (R2+T2) | Radar gain | CLOCs-R vs SECOND |
|---|---|---|---|---|---|---|
| All conditions | 16,372 | 0.453 | 0.414 (0.405 to 0.424) | 0.422 (0.407 to 0.430) | +0.008 (−0.003 to +0.025) | −0.031 |
| Clear | 13,685 | 0.454 | 0.420 (0.412 to 0.427) | 0.425 (0.413 to 0.430) | +0.005 (−0.003 to +0.018) | −0.029 |
| Highway | 2,333 | 0.425 | 0.390 (0.381 to 0.399) | 0.406 (0.386 to 0.416) | +0.016 (−0.001 to +0.034) | −0.019 |
| Urban | 11,352 | 0.467 | 0.430 (0.422 to 0.437) | 0.432 (0.423 to 0.436) | +0.002 (−0.004 to +0.013) | −0.035 |
| Adverse | 2,687 | 0.461 | 0.418 (0.399 to 0.445) | 0.447 (0.404 to 0.470) | +0.029 (−0.004 to +0.071) | −0.014 |
| Night | 1,349 | 0.445 | 0.425 (0.411 to 0.443) | 0.434 (0.412 to 0.443) | +0.009 (−0.008 to +0.032) | −0.011 |
| Rain | 1,338 | 0.479 | 0.421 (0.394 to 0.460) | 0.473 (0.405 to 0.509) | +0.052 (+0.002 to +0.115) | −0.006 |


### By distance, all conditions

| Distance | Cars | SECOND | R0+T2 | R2+T2 | Radar gain |
|---|---|---|---|---|---|
| all | 21,418 | 0.551 | 0.522 (0.514 to 0.530) | 0.532 (0.520 to 0.538) | +0.011 (+0.003 to +0.024) |
| 0-30 m | 3,111 | 0.901 | 0.918 (0.918 to 0.919) | 0.917 (0.916 to 0.919) | −0.001 (−0.002 to +0.000) |
| 30-50 m | 1,895 | 0.645 | 0.664 (0.662 to 0.668) | 0.671 (0.667 to 0.674) | +0.007 (−0.000 to +0.010) |
| 50-100 m | 8,148 | 0.612 | 0.574 (0.568 to 0.583) | 0.577 (0.568 to 0.580) | +0.002 (−0.004 to +0.011) |
| 100-200 m | 8,224 | 0.280 | 0.252 (0.243 to 0.260) | 0.263 (0.243 to 0.273) | +0.010 (−0.002 to +0.030) |
| 50-200 m | 16,372 | 0.453 | 0.414 (0.405 to 0.424) | 0.422 (0.407 to 0.430) | +0.008 (−0.003 to +0.025) |


## Camera removed at test time

Center-distance AP. SECOND alone: holdout 0.440, val 0.453. The network rarely sees candidates without any 2D box during training, so these inputs are unusual for it.

| Model | Split | With camera, beyond 50 m | Camera removed, beyond 50 m | With camera, 100-200 m | Camera removed, 100-200 m |
|---|---|---|---|---|---|
| CLOCs (R0+T2) | holdout | 0.427 (0.422 to 0.433) | 0.424 (0.376 to 0.438) | 0.371 (0.370 to 0.373) | 0.329 (0.255 to 0.348) |
| CLOCs (R0+T2) | val | 0.414 (0.405 to 0.424) | 0.454 (0.433 to 0.460) | 0.252 (0.243 to 0.260) | 0.280 (0.267 to 0.284) |
| CLOCs-R (R2+T2) | holdout | 0.435 (0.434 to 0.436) | 0.431 (0.424 to 0.436) | 0.374 (0.371 to 0.375) | 0.338 (0.318 to 0.351) |
| CLOCs-R (R2+T2) | val | 0.422 (0.407 to 0.430) | 0.461 (0.456 to 0.463) | 0.263 (0.243 to 0.273) | 0.286 (0.277 to 0.291) |


## Latency per frame (median of 519 val frames)

Radar features on one CPU thread in Python; fusion on the GPU; detectors excluded. The fusion forward times are dominated by GPU launch overhead and vary between runs (R0 was timed first and includes warm-up), so only the radar-feature times separate the sets.

| Set | Radar features, ms (95th pct.) | Fusion forward, ms | Total, ms | Parameters |
|---|---|---|---|---|
| R0 | none | 8.12 | 8.1 | 2,143 |
| R1 | 3.2 (5.6) | 3.13 | 6.3 | 2,161 |
| R2 | 19.1 (31.3) | 0.38 | 19.5 | 2,161 |
| R3 | 19.1 (31.6) | 2.48 | 21.6 | 2,179 |
| R4 | 47.3 (81.2) | 2.77 | 50.0 | 2,197 |
| R5 | 20.6 (33.6) | 0.39 | 21.0 | 2,197 |
| R6 | 32.8 (56.2) | 0.45 | 33.2 | 2,197 |
| R7 | 64.3 (109.5) | 2.80 | 67.1 | 2,233 |
