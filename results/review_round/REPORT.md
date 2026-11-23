# CLOCs-R review round: extra measurements

Branch `review-round` (from `master` 0512cd6), 2026-11-23. No detector was retrained, and no stored result was
changed. Code and raw outputs are in `results/review_round/`. Numbers are to four decimals.

## Task 1. Recording-level bootstrap

`results/review_round/cluster_bootstrap.py` (run with the thesis code, which scores L0, LRf and B) computes the
per-frame matches of every model and seed once. It then resamples them in two ways, with 1,000 resamples each,
seeds averaged inside each resample, percentile 95% intervals, and `numpy.random.RandomState(0)` as in
`clocs_r/bootstrap.py`:

- **Scene level:** draws scenes with replacement; identical to the stored bootstrap.
- **Recording level:** draws recordings (`clocs_r.splits.recording`) with replacement and keeps all their scenes.

The point estimate is the APCD difference of the 6-seed means on the whole split.
**Scene (stored)** is the interval in `results/lcr/data/bootstrap_*.json`. **Scene (rerun)** is this run's, and the
reproduction check compares the two. **Recording** is new. LC − L0 had never been bootstrapped, so its scene
interval is from this run.

### `select`: 24 scenes from 11 recordings, 3,409 frames

Scenes per recording: 20220202-105828@Cali 5, 20210401-120439@Jarvis 4, 20210729-162718@Yoda 4, 20210427-130007@Jarvis 2, 20210625-175142@Yoda 2, 20210701-225716@Yoda 2, 20210630-130826@Yoda 1, 20210724-064134@Yoda 1, 20210729-223722@Yoda 1, 20210817-214850@Yoda 1, 20220204-100251@Cali 1.

**Reproduction check (scene level against the stored files):**
- 52 comparison-bins checked.
- Largest differences: model point APCD 0.0e+00, comparison estimate 0.0e+00, interval end 0.0e+00.

| Comparison | Bin (m) | Estimate | Scene (stored) | Scene (rerun) | Recording (new) |
|---|---|---|---|---|---|
| LRd − L0 | 50-100 | +0.0165 | [+0.0071, +0.0252] | [+0.0071, +0.0252] | [+0.0047, +0.0259] |
| LRd − L0 | 100-200 | +0.0100 | [+0.0017, +0.0170] | [+0.0017, +0.0170] | [+0.0038, +0.0148] |
| LRd − L0 | 50-200 | +0.0148 | [+0.0070, +0.0218] | [+0.0070, +0.0218] | [+0.0056, +0.0227] |
| LRd − L0 | all | +0.0071 | [+0.0023, +0.0114] | [+0.0023, +0.0114] | [+0.0031, +0.0134] |
| LRf − L0 | 50-100 | +0.0118 | [-0.0036, +0.0274] | [-0.0036, +0.0274] | [-0.0063, +0.0253] |
| LRf − L0 | 100-200 | +0.0076 | [-0.0065, +0.0208] | [-0.0065, +0.0208] | [-0.0031, +0.0200] |
| LRf − L0 | 50-200 | +0.0124 | [-0.0018, +0.0262] | [-0.0018, +0.0262] | [-0.0042, +0.0234] |
| LRf − L0 | all | +0.0025 | [-0.0047, +0.0100] | [-0.0047, +0.0100] | [-0.0031, +0.0102] |
| LRd − LRf | 50-100 | +0.0047 | [-0.0058, +0.0155] | [-0.0058, +0.0155] | [-0.0012, +0.0132] |
| LRd − LRf | 100-200 | +0.0024 | [-0.0103, +0.0135] | [-0.0103, +0.0135] | [-0.0114, +0.0121] |
| LRd − LRf | 50-200 | +0.0024 | [-0.0078, +0.0128] | [-0.0078, +0.0128] | [-0.0031, +0.0113] |
| LRd − LRf | all | +0.0046 | [-0.0007, +0.0096] | [-0.0007, +0.0096] | [+0.0005, +0.0101] |
| LRd − L | 50-100 | +0.0154 | [+0.0058, +0.0241] | [+0.0058, +0.0241] | [+0.0036, +0.0252] |
| LRd − L | 100-200 | +0.0110 | [+0.0032, +0.0174] | [+0.0032, +0.0174] | [+0.0051, +0.0154] |
| LRd − L | 50-200 | +0.0114 | [+0.0032, +0.0185] | [+0.0032, +0.0185] | [+0.0021, +0.0200] |
| LRd − L | all | -0.0020 | [-0.0103, +0.0050] | [-0.0103, +0.0050] | [-0.0106, +0.0045] |
| LC − L | 50-100 | +0.0220 | [-0.0048, +0.0522] | [-0.0048, +0.0522] | [+0.0016, +0.0438] |
| LC − L | 100-200 | +0.0045 | [-0.0189, +0.0298] | [-0.0189, +0.0298] | [-0.0184, +0.0293] |
| LC − L | 50-200 | +0.0168 | [-0.0055, +0.0424] | [-0.0055, +0.0424] | [-0.0017, +0.0379] |
| LC − L | all | +0.0058 | [-0.0036, +0.0186] | [-0.0036, +0.0186] | [-0.0026, +0.0196] |
| LC − L0 | 50-100 | +0.0232 | not stored | [-0.0034, +0.0541] | [+0.0030, +0.0445] |
| LC − L0 | 100-200 | +0.0034 | not stored | [-0.0208, +0.0286] | [-0.0199, +0.0281] |
| LC − L0 | 50-200 | +0.0202 | not stored | [-0.0025, +0.0463] | [+0.0018, +0.0409] |
| LC − L0 | all | +0.0149 | not stored | [+0.0049, +0.0279] | [+0.0062, +0.0296] |
| B − L | 50-100 | +0.0340 | [+0.0134, +0.0564] | [+0.0134, +0.0564] | [+0.0142, +0.0542] |
| B − L | 100-200 | +0.0210 | [+0.0014, +0.0410] | [+0.0014, +0.0410] | [+0.0068, +0.0397] |
| B − L | 50-200 | +0.0294 | [+0.0121, +0.0480] | [+0.0121, +0.0480] | [+0.0139, +0.0476] |
| B − L | all | +0.0150 | [+0.0069, +0.0254] | [+0.0069, +0.0254] | [+0.0063, +0.0283] |
| B − LC | 50-100 | +0.0120 | [+0.0024, +0.0207] | [+0.0024, +0.0207] | [+0.0082, +0.0167] |
| B − LC | 100-200 | +0.0165 | [+0.0076, +0.0260] | [+0.0076, +0.0260] | [+0.0051, +0.0292] |
| B − LC | 50-200 | +0.0126 | [+0.0043, +0.0204] | [+0.0043, +0.0204] | [+0.0082, +0.0181] |
| B − LC | all | +0.0092 | [+0.0051, +0.0131] | [+0.0051, +0.0131] | [+0.0068, +0.0113] |
| LCR − LC | 50-100 | +0.0012 | [-0.0028, +0.0056] | [-0.0028, +0.0056] | [-0.0019, +0.0055] |
| LCR − LC | 100-200 | +0.0029 | [+0.0001, +0.0067] | [+0.0001, +0.0067] | [-0.0003, +0.0064] |
| LCR − LC | 50-200 | +0.0010 | [-0.0022, +0.0047] | [-0.0022, +0.0047] | [-0.0014, +0.0038] |
| LCR − LC | all | +0.0002 | [-0.0036, +0.0034] | [-0.0036, +0.0034] | [-0.0017, +0.0019] |
| LCR − L | 50-100 | +0.0232 | [-0.0020, +0.0515] | [-0.0020, +0.0515] | [+0.0015, +0.0479] |
| LCR − L | 100-200 | +0.0074 | [-0.0156, +0.0323] | [-0.0156, +0.0323] | [-0.0140, +0.0323] |
| LCR − L | 50-200 | +0.0178 | [-0.0036, +0.0429] | [-0.0036, +0.0429] | [-0.0018, +0.0415] |
| LCR − L | all | +0.0060 | [-0.0036, +0.0183] | [-0.0036, +0.0183] | [-0.0027, +0.0206] |
| LCR − LRd | 50-100 | +0.0078 | [-0.0128, +0.0326] | [-0.0128, +0.0326] | [-0.0049, +0.0253] |
| LCR − LRd | 100-200 | -0.0037 | [-0.0240, +0.0210] | [-0.0240, +0.0210] | [-0.0243, +0.0225] |
| LCR − LRd | 50-200 | +0.0064 | [-0.0117, +0.0278] | [-0.0117, +0.0278] | [-0.0067, +0.0246] |
| LCR − LRd | all | +0.0080 | [+0.0002, +0.0178] | [+0.0002, +0.0178] | [+0.0026, +0.0188] |
| LCR − B | 50-100 | -0.0108 | [-0.0169, -0.0029] | [-0.0169, -0.0029] | [-0.0171, -0.0042] |
| LCR − B | 100-200 | -0.0136 | [-0.0213, -0.0060] | [-0.0213, -0.0060] | [-0.0234, -0.0044] |
| LCR − B | 50-200 | -0.0116 | [-0.0174, -0.0041] | [-0.0174, -0.0041] | [-0.0179, -0.0047] |
| LCR − B | all | -0.0090 | [-0.0126, -0.0057] | [-0.0126, -0.0057] | [-0.0108, -0.0071] |
| LCR-D − LCR | 50-100 | -0.0041 | [-0.0183, +0.0085] | [-0.0183, +0.0085] | [-0.0165, +0.0046] |
| LCR-D − LCR | 100-200 | +0.0056 | [-0.0091, +0.0193] | [-0.0091, +0.0193] | [-0.0084, +0.0194] |
| LCR-D − LCR | 50-200 | -0.0020 | [-0.0145, +0.0096] | [-0.0145, +0.0096] | [-0.0128, +0.0070] |
| LCR-D − LCR | all | -0.0002 | [-0.0060, +0.0052] | [-0.0060, +0.0052] | [-0.0059, +0.0029] |
| LCR-D − L | 50-100 | +0.0191 | [+0.0008, +0.0377] | [+0.0008, +0.0377] | [+0.0019, +0.0376] |
| LCR-D − L | 100-200 | +0.0130 | [-0.0048, +0.0322] | [-0.0048, +0.0322] | [-0.0014, +0.0293] |
| LCR-D − L | 50-200 | +0.0158 | [-0.0006, +0.0318] | [-0.0006, +0.0318] | [+0.0003, +0.0332] |
| LCR-D − L | all | +0.0058 | [-0.0034, +0.0150] | [-0.0034, +0.0150] | [-0.0017, +0.0165] |

**Where the verdict changes:**
- **Scene interval excludes 0, recording interval includes 0:** LCR − LC at 100-200 m.
- **Scene interval includes 0, recording interval excludes 0:** LRd − LRf, all ranges, LC − L at 50-100 m, LC − L0 at 50-100 m, LC − L0 at 50-200 m, LCR − L at 50-100 m, LCR-D − L at 50-200 m.
- **Width:** the recording interval is narrower than the scene interval in 42 of 56 comparison-bins.

**Caveat:** with only 11 recordings, some holding a single scene, a percentile bootstrap over
recordings is itself unreliable and tends to understate the spread. Where a recording interval is narrower
than the scene interval, that does not show that recordings add no correlation. Read the second list as weak
evidence, and the first list as the cases the paper should not call significant without qualification.

### val: 25 scenes from 8 recordings, 5,181 frames

Scenes per recording: 20210728-063719@Yoda 9, 20210401-074452@Jarvis 4, 20210401-074958@Jarvis 4, 20210401-081346@Jarvis 3, 20211006-112855@Sogun 2, 20210401-085844@Jarvis 1, 20210901-194123@Yoda 1, 20210920-203858@Sogun 1.

**Reproduction check (scene level against the stored files):**
- 52 comparison-bins checked.
- Largest differences: model point APCD 0.0e+00, comparison estimate 0.0e+00, interval end 0.0e+00.

| Comparison | Bin (m) | Estimate | Scene (stored) | Scene (rerun) | Recording (new) |
|---|---|---|---|---|---|
| LRd − L0 | 50-100 | +0.0049 | [-0.0088, +0.0206] | [-0.0088, +0.0206] | [+0.0002, +0.0171] |
| LRd − L0 | 100-200 | +0.0129 | [+0.0078, +0.0179] | [+0.0078, +0.0179] | [+0.0112, +0.0194] |
| LRd − L0 | 50-200 | +0.0133 | [+0.0049, +0.0228] | [+0.0049, +0.0228] | [+0.0092, +0.0253] |
| LRd − L0 | all | +0.0106 | [+0.0029, +0.0190] | [+0.0029, +0.0190] | [+0.0079, +0.0174] |
| LRf − L0 | 50-100 | -0.0113 | [-0.0217, +0.0037] | [-0.0217, +0.0037] | [-0.0158, +0.0155] |
| LRf − L0 | 100-200 | +0.0003 | [-0.0065, +0.0060] | [-0.0065, +0.0060] | [-0.0133, +0.0056] |
| LRf − L0 | 50-200 | -0.0049 | [-0.0128, +0.0039] | [-0.0128, +0.0039] | [-0.0084, +0.0082] |
| LRf − L0 | all | -0.0102 | [-0.0185, -0.0012] | [-0.0185, -0.0012] | [-0.0185, +0.0069] |
| LRd − LRf | 50-100 | +0.0162 | [+0.0017, +0.0248] | [+0.0017, +0.0248] | [-0.0041, +0.0178] |
| LRd − LRf | 100-200 | +0.0126 | [+0.0068, +0.0191] | [+0.0068, +0.0191] | [+0.0071, +0.0287] |
| LRd − LRf | 50-200 | +0.0182 | [+0.0104, +0.0238] | [+0.0104, +0.0238] | [+0.0092, +0.0279] |
| LRd − LRf | all | +0.0208 | [+0.0115, +0.0278] | [+0.0115, +0.0278] | [+0.0073, +0.0288] |
| LRd − L | 50-100 | +0.0036 | [-0.0106, +0.0189] | [-0.0106, +0.0189] | [-0.0013, +0.0152] |
| LRd − L | 100-200 | +0.0016 | [-0.0042, +0.0079] | [-0.0042, +0.0079] | [-0.0088, +0.0117] |
| LRd − L | 50-200 | -0.0040 | [-0.0146, +0.0050] | [-0.0146, +0.0050] | [-0.0113, +0.0055] |
| LRd − L | all | -0.0275 | [-0.0393, -0.0176] | [-0.0393, -0.0176] | [-0.0351, -0.0077] |
| LC − L | 50-100 | -0.0225 | [-0.0366, -0.0092] | [-0.0366, -0.0092] | [-0.0318, +0.0040] |
| LC − L | 100-200 | -0.0052 | [-0.0231, +0.0135] | [-0.0231, +0.0135] | [-0.0344, +0.0280] |
| LC − L | 50-200 | -0.0191 | [-0.0312, -0.0071] | [-0.0312, -0.0071] | [-0.0317, +0.0103] |
| LC − L | all | -0.0154 | [-0.0272, -0.0039] | [-0.0272, -0.0039] | [-0.0252, +0.0091] |
| LC − L0 | 50-100 | -0.0211 | not stored | [-0.0350, -0.0075] | [-0.0303, +0.0051] |
| LC − L0 | 100-200 | +0.0060 | not stored | [-0.0087, +0.0230] | [-0.0102, +0.0348] |
| LC − L0 | 50-200 | -0.0018 | not stored | [-0.0123, +0.0107] | [-0.0097, +0.0245] |
| LC − L0 | all | +0.0227 | not stored | [+0.0115, +0.0348] | [+0.0167, +0.0390] |
| B − L | 50-100 | -0.0007 | [-0.0118, +0.0079] | [-0.0118, +0.0079] | [-0.0071, +0.0108] |
| B − L | 100-200 | +0.0070 | [-0.0057, +0.0199] | [-0.0057, +0.0199] | [-0.0078, +0.0312] |
| B − L | 50-200 | -0.0027 | [-0.0130, +0.0057] | [-0.0130, +0.0057] | [-0.0123, +0.0138] |
| B − L | all | -0.0024 | [-0.0109, +0.0052] | [-0.0109, +0.0052] | [-0.0106, +0.0113] |
| B − LC | 50-100 | +0.0218 | [+0.0149, +0.0270] | [+0.0149, +0.0270] | [+0.0049, +0.0258] |
| B − LC | 100-200 | +0.0122 | [+0.0055, +0.0189] | [+0.0055, +0.0189] | [+0.0025, +0.0273] |
| B − LC | 50-200 | +0.0164 | [+0.0115, +0.0200] | [+0.0115, +0.0200] | [+0.0033, +0.0198] |
| B − LC | all | +0.0130 | [+0.0079, +0.0171] | [+0.0079, +0.0171] | [+0.0002, +0.0169] |
| LCR − LC | 50-100 | +0.0036 | [-0.0030, +0.0111] | [-0.0030, +0.0111] | [-0.0025, +0.0049] |
| LCR − LC | 100-200 | -0.0010 | [-0.0053, +0.0029] | [-0.0053, +0.0029] | [-0.0054, +0.0061] |
| LCR − LC | 50-200 | +0.0007 | [-0.0042, +0.0051] | [-0.0042, +0.0051] | [-0.0027, +0.0028] |
| LCR − LC | all | +0.0008 | [-0.0025, +0.0039] | [-0.0025, +0.0039] | [-0.0071, +0.0030] |
| LCR − L | 50-100 | -0.0189 | [-0.0362, -0.0030] | [-0.0362, -0.0030] | [-0.0286, +0.0066] |
| LCR − L | 100-200 | -0.0062 | [-0.0271, +0.0145] | [-0.0271, +0.0145] | [-0.0377, +0.0323] |
| LCR − L | 50-200 | -0.0184 | [-0.0332, -0.0046] | [-0.0332, -0.0046] | [-0.0336, +0.0126] |
| LCR − L | all | -0.0146 | [-0.0270, -0.0023] | [-0.0270, -0.0023] | [-0.0260, +0.0078] |
| LCR − LRd | 50-100 | -0.0225 | [-0.0331, -0.0100] | [-0.0331, -0.0100] | [-0.0316, -0.0025] |
| LCR − LRd | 100-200 | -0.0078 | [-0.0240, +0.0092] | [-0.0240, +0.0092] | [-0.0296, +0.0234] |
| LCR − LRd | 50-200 | -0.0144 | [-0.0221, -0.0041] | [-0.0221, -0.0041] | [-0.0231, +0.0098] |
| LCR − LRd | all | +0.0129 | [+0.0048, +0.0218] | [+0.0048, +0.0218] | [+0.0064, +0.0242] |
| LCR − B | 50-100 | -0.0182 | [-0.0260, -0.0085] | [-0.0260, -0.0085] | [-0.0225, -0.0019] |
| LCR − B | 100-200 | -0.0132 | [-0.0223, -0.0046] | [-0.0223, -0.0046] | [-0.0302, +0.0014] |
| LCR − B | 50-200 | -0.0157 | [-0.0220, -0.0091] | [-0.0220, -0.0091] | [-0.0210, -0.0008] |
| LCR − B | all | -0.0122 | [-0.0175, -0.0066] | [-0.0175, -0.0066] | [-0.0156, -0.0014] |
| LCR-D − LCR | 50-100 | +0.0097 | [-0.0029, +0.0242] | [-0.0029, +0.0242] | [+0.0013, +0.0274] |
| LCR-D − LCR | 100-200 | +0.0022 | [-0.0149, +0.0204] | [-0.0149, +0.0204] | [-0.0283, +0.0240] |
| LCR-D − LCR | 50-200 | +0.0076 | [-0.0030, +0.0184] | [-0.0030, +0.0184] | [-0.0082, +0.0204] |
| LCR-D − LCR | all | +0.0061 | [-0.0016, +0.0149] | [-0.0016, +0.0149] | [-0.0059, +0.0147] |
| LCR-D − L | 50-100 | -0.0091 | [-0.0284, +0.0123] | [-0.0284, +0.0123] | [-0.0165, +0.0122] |
| LCR-D − L | 100-200 | -0.0040 | [-0.0136, +0.0048] | [-0.0136, +0.0048] | [-0.0171, +0.0086] |
| LCR-D − L | 50-200 | -0.0108 | [-0.0225, +0.0019] | [-0.0225, +0.0019] | [-0.0157, +0.0066] |
| LCR-D − L | all | -0.0085 | [-0.0191, +0.0038] | [-0.0191, +0.0038] | [-0.0142, +0.0087] |

**Where the verdict changes:**
- **Scene interval excludes 0, recording interval includes 0:** LRf − L0, all ranges, LRd − LRf at 50-100 m, LC − L at 50-100 m, LC − L at 50-200 m, LC − L, all ranges, LC − L0 at 50-100 m, LCR − L at 50-100 m, LCR − L at 50-200 m, LCR − L, all ranges, LCR − LRd at 50-200 m, LCR − B at 100-200 m.
- **Scene interval includes 0, recording interval excludes 0:** LRd − L0 at 50-100 m, LCR-D − LCR at 50-100 m.
- **Width:** the recording interval is narrower than the scene interval in 15 of 56 comparison-bins.

**Caveat:** with only 8 recordings, some holding a single scene, a percentile bootstrap over
recordings is itself unreliable and tends to understate the spread. Where a recording interval is narrower
than the scene interval, that does not show that recordings add no correlation. Read the second list as weak
evidence, and the first list as the cases the paper should not call significant without qualification.

## Task 2. Is SECOND more reliable on its own training scenes?

`results/review_round/second_reliability.py` scores the cached candidates with the `eval_detectors` code, and
reproduces the stored `select` and val numbers. The largest differences are 0.0e+00 for SECOND's APCD and 0.0e+00 for
Cascade R-CNN's recall.

**SECOND alone, APCD** (`fit` is in-sample: SECOND was trained on it):

| Split | Frames | 0–30 | 30–50 | 50–100 | 100–200 | 50–200 |
|---|---|---|---|---|---|---|
| fit | 17,993 | 0.9585 | 0.8552 | 0.7528 | 0.4694 | 0.6961 |
| select | 3,409 | 0.9072 | 0.7797 | 0.6277 | 0.4052 | 0.5639 |
| val | 5,181 | 0.8971 | 0.6206 | 0.5992 | 0.2841 | 0.4455 |

**Reliability of SECOND's candidates beyond 50 m.** These are the candidates after BEV NMS, as in evaluation
(region and camera view, trucks and buses removed, NMS at IoU 0.1). "True" is the share with a scored car whose
BEV centre is within 2 m; n is the number of candidates.

| Score | fit: true (n) | select: true (n) | val: true (n) |
|---|---|---|---|
| [0.05, 0.1) | 0.1142 (83,252) | 0.1577 (15,145) | 0.2767 (14,644) |
| [0.1, 0.2) | 0.2312 (32,503) | 0.3235 (6,245) | 0.5275 (5,463) |
| [0.2, 0.3) | 0.4598 (11,493) | 0.5842 (2,564) | 0.7931 (2,450) |
| [0.3, 0.5) | 0.8113 (11,754) | 0.7575 (2,223) | 0.9262 (1,937) |
| [0.5, 0.7) | 0.9699 (7,042) | 0.8945 (1,270) | 0.9650 (1,142) |
| [0.7, 1.0] | 0.9984 (13,274) | 0.9459 (1,756) | 0.9936 (1,566) |

**Cascade R-CNN (trained on `fit`), candidate recall@0.5 for cars at 100–200 m:**

| fit (in-sample) | select | val |
|---|---|---|
| 0.9111 | 0.3531 | 0.2271 |

## Task 3. Size and speed

**Hardware:**
- GPU: GRID A100X-10C (an NVIDIA vGPU slice of an A100: 10240 MiB, driver 580.159.03).
- CPU: AMD EPYC-Milan Processor, 8 cores available.
- PyTorch 2.0.1+cu118, CUDA 11.8, cuDNN 8700, 8 CPU threads.
- Nothing else ran on the machine during the timing.

Batch size 1, on 200 val frames spread evenly over the split, after 20 warm-up frames. CUDA was
synchronized before and after each timed step. Code: `results/review_round/timing.py`.

| Step | Runs on | Median (ms) | 90th pct (ms) |
|---|---|---|---|
| SECOND input | CPU | 6.1 | 8.1 |
| SECOND network | GPU | 11.7 | 13.2 |
| Cascade R-CNN at 1.5x | image read and resize on CPU, network on GPU | 110.9 | 113.9 |
| D2 input | CPU | 21.5 | 26.2 |
| D2 network | GPU | 4.5 | 4.8 |
| D2 output filter: score >= 0.01, BEV NMS | CPU | 43.1 | 58.5 |
| LCR cells: camera pairs with K40, radar pairs, reserved cells | CPU | 8.4 | 11.0 |
| LCR fusion network: cells, max per candidate, residual, sigmoid | GPU | 8.7 | 9.6 |
| BEV NMS of the fused candidates, IoU 0.1, at most 100 | CPU | 32.5 | 37.5 |

Notes on the steps:
- **SECOND and D2, input:** includes reading the frame's files from disk.
- **SECOND, network:** ends at the top 2,000 boxes before NMS, as the cached candidates are made.
- **Cascade R-CNN:** one call to MMDetection's `inference_detector` (read the JPEG, resize to [1920, 1056], normalize, network, post-processing).
- **LCR cells:** start from the three detectors' outputs, already in memory.
- **Cell check:** the timed code's cells were checked equal to the cached LCR cells on 220 frames.

**Parameters.** "Parameters" counts the trainable parameters of the model as built and loaded. "Checkpoint
elements" also counts BatchNorm running statistics.

| Model | Parameters | Checkpoint elements |
|---|---|---|
| SECOND | 5,298,900 | 5,305,839 |
| D2 (radar PointPillars) | 4,815,060 | 4,821,097 |
| Cascade R-CNN (ResNet-50 FPN) | 69,151,841 | 69,205,014 |
| Clutter MLP (12-64-64-1) | 5,057 | 5,057 |
| LCR fusion network | 2,179 | 2,179 |

## Task 4. L0 and LRf on `master`

- **Copied unchanged** (byte-identical) from thesis `results/next_round/phase7/`: `experts.json` as
  `results/lcr/data/experts_nc.json` (`select`) and `val_experts.json` as `results/lcr/data/val_experts_nc.json`
  (val).
- **Contents:** both hold R0+T2+NC (L0) and R2+T2+NC (LRf), seeds 1–6.
- **Check:** their 6-seed means are the paper's values. At 50–200 m: L0 0.5605 on `select` and 0.4282 on val; LRf
  0.5729 and 0.4233.
- **Documentation:** one line added to `docs/MODELS.md` ("Stored results", under "Radar as features without a
  camera").
- **Committed** locally on `review-round`; not pushed.

## Task 5. Pairing-radius sensitivity (post-hoc check, not a selection)

LRd (R0+T2+RP) was retrained on `fit` with radar pairing radius r, affinity 1 − d / r, seeds 1–3. Everything
else is unchanged (`results/review_round/radius_lrd.py`; the code's default stays 2 m). Each model was scored on
`select`, then once on val from the saved networks. The 2 m retrain reproduces the stored LRd exactly: for
each of seeds 1–3 the difference is 0.0 in every bin and condition, so training is deterministic here.

Values are APCD, mean ± SD over seeds 1–3.

| Radius | Split | 0-30 | 30-50 | 50-100 | 100-200 | 50-200 |
|---|---|---|---|---|---|---|
| 1.0 m | select | 0.9081 ± 0.0002 | 0.7759 ± 0.0006 | 0.6423 ± 0.0007 | 0.4158 ± 0.0011 | 0.5747 ± 0.0005 |
| 1.0 m | val | 0.8917 ± 0.0002 | 0.6197 ± 0.0006 | 0.6022 ± 0.0005 | 0.2837 ± 0.0020 | 0.4393 ± 0.0023 |
| 2.0 m | select | 0.9077 ± 0.0001 | 0.7759 ± 0.0010 | 0.6434 ± 0.0014 | 0.4166 ± 0.0024 | 0.5754 ± 0.0013 |
| 2.0 m | val | 0.8916 ± 0.0009 | 0.6214 ± 0.0005 | 0.6030 ± 0.0003 | 0.2864 ± 0.0007 | 0.4413 ± 0.0019 |
| 3.0 m | select | 0.9076 ± 0.0001 | 0.7750 ± 0.0008 | 0.6428 ± 0.0014 | 0.4150 ± 0.0008 | 0.5747 ± 0.0014 |
| 3.0 m | val | 0.8919 ± 0.0004 | 0.6190 ± 0.0011 | 0.6020 ± 0.0006 | 0.2855 ± 0.0013 | 0.4408 ± 0.0019 |

**Reproduction check:** the stored LRd (radius 2 m), mean of seeds 1–3, against this run's 2 m:

| Split | 0-30 | 30-50 | 50-100 | 100-200 | 50-200 |
|---|---|---|---|---|---|
| select, stored | 0.9077 | 0.7759 | 0.6434 | 0.4166 | 0.5754 |
| select, rerun | 0.9077 | 0.7759 | 0.6434 | 0.4166 | 0.5754 |
| val, stored | 0.8916 | 0.6214 | 0.6030 | 0.2864 | 0.4413 |
| val, rerun | 0.8916 | 0.6214 | 0.6030 | 0.2864 | 0.4413 |

## Checks against the paper draft

- **Table VI, B's gain over LC on val, 50–200 m,** quoted as "+0.016 [+0.012, +0.020]": the stored value is
  +0.0164 [+0.0115, +0.0200]. Matches.
- **Text, "B beats CLOCs-R by 0.016 [0.009, 0.022]":** this is B − LCR on val at 50–200 m. The stored LCR − B is
  -0.0157 [-0.0220, -0.0091], so B − LCR = +0.0157 [+0.0091, +0.0220]. Matches. The
  two quotes differ only because they compare B with different models.
- **Typos:** the abstract begins "BBeyond", and Table V's header reads "50-−200 m".

## What did not go as planned, and what to know

- **Caches rebuilt.** The base CLOCs cells (`runs/next_round/phase1/cells_in/{fit,select,val}_base.pkl`) had been
  deleted in a disk cleanup earlier on 2026-11-23, with the user's approval. They were rebuilt with `cache_cells`
  before tasks 1 and 5. A test rebuild of `select` that morning was byte-identical to the deleted file, and all three
  files came back at exactly their stored byte sizes. The reproduction checks of tasks 1 and 5 confirm the rebuilt
  cells give the stored results.
- **New cache files, none changed.**
  - The timing script's cell check wrote one new pair cache, `cells_in/val_rp_lcr_d2_dets_min0.05_m2.0.pkl` (12 MB). It
    is byte-identical to the existing `val_rp_phase7_d2_dets_min0.05_m2.0.pkl`, under another name.
  - Task 5's pair caches went to their own folders (`runs/review_round/cells_r*`).
  - No existing result, run output or cache was modified.
- **Disk.**
  - Free space reached 5.0 GB during the run. The cause was not this work: GNOME's file indexer
    (`tracker-miner-fs-3`) fell into an error loop over editor temporary files and wrote about 4.7 GB of warnings
    to `/var/log/syslog` and the system journal.
  - The indexer was stopped (`systemctl --user stop tracker-miner-fs-3`), and the growth stopped.
  - The user then masked the indexer and cleared the logs (`journalctl --vacuum-size=200M`, `truncate` of `syslog`),
    bringing free space to 9.3 GB before the timing run.
- **Parallel bootstrap.** Task 1's val bootstrap ran in parallel with `select`, in its own process, to save time.
  The results do not depend on this.
- **CPU-only steps.** In task 3, BEV NMS (`clocs_r.metrics.nms_bev`) is plain NumPy on the CPU. It computes rotated
  IoU between up to 1,000 boxes, so it dominates the post-detector time. A GPU NMS would be much faster, but none was
  used in any reported result.

