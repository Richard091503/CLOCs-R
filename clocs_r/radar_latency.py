"""Per-frame time to compute radar features: accumulate sweeps, compensate Doppler, gate and summarize (Section II-F).

Runs on one CPU thread with sequences already in memory, so disk time is left out. The fusion forward pass is timed in train_fusion.

Usage: python -m clocs_r.radar_latency --delta0 0.5 --kappa 0.02 --out results/clocs_radar/radar_latency.json
"""
import argparse
import json
import pathlib
import time

import numpy as np

from clocs_r import cache_cells, radar_features, splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta0", type=float, required=True)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--sweeps", type=int, default=cache_cells.SWEEPS)
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items = splits.frames("val")
    items = items[:: max(1, len(items) // args.frames)]
    times = []
    for seq, frame in items:
        with np.load(pathlib.Path("runs/candidates3d") / seq.name / f"{frame:07d}.npz") as z:
            boxes = z["boxes"]
        _ = seq.radar  # load the sequence's sweeps before the clock starts
        t0 = time.perf_counter()
        pts, vel = seq.radar_accumulated(frame, args.sweeps)
        radar_features.candidate_features(boxes, pts, vel, args.delta0, args.kappa)
        times.append(time.perf_counter() - t0)
    result = dict(vars(args), frames=len(times), median_ms=float(np.median(times) * 1000), p95_ms=float(np.percentile(times, 95) * 1000))
    print(result)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
