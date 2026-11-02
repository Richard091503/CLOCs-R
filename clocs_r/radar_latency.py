"""Time each feature set adds per frame: computing its radar features, on one CPU thread, and the fusion forward pass, on the GPU (Section II-F).

Run with nothing else on the machine. Sequences are already in memory, so disk time is left out, and the detectors are not included.

Usage: python -m clocs_r.radar_latency --delta0 0.5 --kappa 0.02 --out results/clocs_radar/ablation/latency.json
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from clocs_r import cache_cells, fusion, radar_features, splits, train_fusion


def feature_times(items, sets, delta0, kappa, sweeps):
    times = {name: [] for name in sets}
    for seq, frame in items:
        with np.load(pathlib.Path("runs/candidates3d") / seq.name / f"{frame:07d}.npz") as z:
            boxes = z["boxes"]
        _ = seq.radar  # load the sequence's sweeps before the clock starts
        for name in sets:
            names = radar_features.FEATURE_SETS[name]
            t0 = time.perf_counter()
            if names:
                pts, vel = seq.radar_accumulated(frame, sweeps)
                radar_features.candidate_features(boxes, pts, vel, delta0, kappa, names)
            times[name].append(time.perf_counter() - t0)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta0", type=float, required=True)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--sweeps", type=int, default=cache_cells.SWEEPS)
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--sets", default="R0,R1,R2,R3,R4,R5,R6,R7")
    ap.add_argument("--fusion_dir", default="runs/fusion")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sets = args.sets.split(",")
    torch.set_num_threads(1)
    device = torch.device("cuda")
    items = splits.frames("val")
    items = items[:: max(1, len(items) // args.frames)]
    features = feature_times(items, sets, args.delta0, args.kappa, args.sweeps)
    base, radar = cache_cells.load("val", "runs/cells", args.delta0, args.kappa, args.sweeps)
    result = dict(vars(args), frames=len(items), sets={})
    for name in sets:
        idx = [radar_features.FEATURES.index(f) for f in radar_features.FEATURE_SETS[name]]
        net = fusion.FusionNet(len(idx)).to(device)
        net.load_state_dict(torch.load(f"{args.fusion_dir}/{name}_s1/fusion.pth", map_location=device)["model"])
        forward = train_fusion.measure_latency(net, base, radar, idx, device)
        feats = np.array(features[name]) * 1000
        result["sets"][name] = dict(features_median_ms=float(np.median(feats)), features_p95_ms=float(np.percentile(feats, 95)), fusion_median_ms=forward,
                                    params=sum(p.numel() for p in net.parameters()))
        print(name, {k: round(v, 2) for k, v in result["sets"][name].items()}, flush=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
