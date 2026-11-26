# Qualitative figure and event counts (2026-11-26)

Everything here is made by `python -m clocs_r.make_qualitative` (about 20 minutes on one GPU) from the cached SECOND
candidates, Cascade R-CNN boxes, D2 boxes and pair cells (`runs/lcr/`) and the stored networks in
`results/lcr/weights/`: LC = `R0+T2+K40_s1.pth`, CLOCs-R = `R0+T2+K40+RC_s1.pth` (code name LCR). Nothing was trained,
and no other file was changed. Redrawing only the figure takes seconds:
`python -m clocs_r.make_qualitative figure [--bev 20] [--pick radar_rescue=4]`.

| File | Contents |
|---|---|
| `reproduction.json` | APCD of SECOND, LC and CLOCs-R (seed 1) from this script against the stored results: identical (difference 0.000000) on select and val, all bins |
| `thresholds.json` | step 1: the F1-maximizing thresholds on select, and F1, precision and recall at them on select and val |
| `events.json` | step 2: definitions, per-split counts, winning-cell shares, a check after NMS, and one record per event |
| `examples.json` | step 3: the rule, and per event type the chosen val example and four alternates |
| `qualitative.pdf`, `.png` | step 4: the figure, BEV window 40 m as specified (PNG at 300 dpi) |
| `qualitative_bev20.pdf`, `.png` | the same figure with a 20 m BEV window (a variant, not in the prompt) |
| `*_figure.json` | which example each row shows and how many Cascade boxes were drawn |

## Results

Thresholds (select, cars at 50–200 m, 2 m centre distance): SECOND 0.197, LC 0.132, CLOCs-R 0.136.

| | select | val |
|---|---|---|
| cars at 50–200 m | 9,634 | 16,372 |
| with a matched SECOND candidate | 9,004 | 15,137 |
| radar rescue | 18 | 16 |
| camera suppression (camera cell / reserved cell) | 383 (337 / 46) | 636 (502 / 134) |
| no proposal | 0 | 163 |
| winning cell, 50–100 m: camera / radar / reserved | 89.6 / 8.3 / 2.1% | 82.9 / 14.6 / 2.5% |
| winning cell, 100–200 m: camera / radar / reserved | 81.0 / 10.6 / 8.4% | 58.8 / 11.3 / 29.9% |

Findings to keep in mind when writing about the figure:
- **Radar rescues are rare and come from a lower threshold, not a higher score.** In 33 of 34 rescues (both splits)
  CLOCs-R's score is *below* SECOND's (val medians 0.186 to 0.143). They cross only because CLOCs-R's threshold is 0.136
  against SECOND's 0.197. In all 34, CLOCs-R is above LC (val median 0.111 to 0.143): the radar cell lifts the score
  relative to the camera-only network. The figure prints the LC score for this reason.
- Camera suppressions outnumber radar rescues about 40 to 1 on val. In all 1,019, CLOCs-R is below SECOND.
- No proposal is 0 on select: 630 far cars there have no SECOND candidate within 2 m, but D2's best score near any of
  them is 0.095, under the 0.1 floor. On val, 163 of 1,235 such cars have a D2 box ≥ 0.1.
- Check after NMS (not part of the event rule): for 15 of 16 val rescues, CLOCs-R has a final detection at threshold
  within 2 m and SECOND has none. For 617 of 636 val suppressions, SECOND has one and CLOCs-R has none (36 still have a
  CLOCs-R detection from another candidate).
- Example (a), the rule's radar rescue (val, urban, 62 m), is a car mostly hidden behind the lead vehicle. In the crop
  the box sits on the lead car's rear window. Three of its four alternates are the same situation; alternate 4 (frame
  8275, 64 m) is partly hidden by a pole. The caption should say so. The val rescues come from only 2 recordings and 14
  of 16 are urban.

## Choices the prompt did not fix

- Final detections for the thresholds: evaluation mask and BEV NMS 0.1, as every APCD in the paper; detections and
  cars each count in 50–200 m by their own range, as in `metrics.evaluate`.
- Matching a car to "its" SECOND candidate before NMS: candidates in descending SECOND score each take the nearest
  unmatched car within 2 m (`metrics._match`); only scored candidates (evaluation mask) take part. "No proposal" uses the
  same candidate set.
- Counts: LiDAR points inside the car's 3D box enlarged by 0.5 m on every side; radar returns of the last 5 sweeps
  with Doppler motion correction (D2's input) inside its BEV box enlarged by 0.5 m (the clutter label's rule). The BEV
  panel shows the same 5 sweeps.
- Car width: the projected 3D box, clipped to the image. Every val event passes the 4 px test (the narrowest is 14.7 px).
- Step 3's median is over all val events of the type; alternates skip events within 10 frames of an already listed one
  in the same sequence, so they are not the same moment.
- Cascade boxes are drawn after a 2D NMS at 0.5. The cache keeps near-duplicates (export NMS 0.95, for CLOCs), which
  drew as stacks of boxes. Scores > 0.05, as asked.
- The D2 box drawn is the one in the winning radar cell for a rescue, otherwise the highest-scoring D2 box within 2 m.
- Score text shows LC as well as SECOND and CLOCs-R (see above), and the three thresholds are printed under the legend.
- A white edge is drawn around black outlines so that they read on dark images and over one another. Text is 6.5 pt at
  the final size; fonts are Liberation Sans as TrueType (Type 42). PDF metadata (Creator, Producer, Author, Title,
  CreationDate) is removed. The camera crops are decoded pixels, so no EXIF is embedded.
- With a 40 m BEV window, a car is about 6 × 2.5 pt on the page, so the solid, dashed and dotted outlines merge. The
  20 m variant keeps them apart.
