"""Scene-by-scene comparison on val of the tuned model, its no-radar version and SECOND: center-distance AP beyond 50 m, seed means.

Usage, from the repo root: python results/clocs_radar/tuning/compare_scenes.py
"""
import json
import pathlib

import numpy as np
import torch

from clocs_r import cache_cells, fusion, metrics, train_fusion

OUT = pathlib.Path(__file__).parent / "compare_scenes.json"
GATE = (0.5, 0.02, 3)
device = torch.device("cuda")
MIN_CARS = 50


def scene_scores(net, base, radar, idx):
    records = [(it["gt"], b, s, it["seq"]) for it, b, s in train_fusion.predict(net, base, radar, idx, device)]
    out = {}
    for scene in sorted({r[3] for r in records}):
        v = metrics.evaluate([r[:3] for r in records if r[3] == scene], bins=[train_fusion.FAR])["50-200"]
        out[scene] = dict(n_gt=v["n_gt"], APCD=v["APCD"])
    return out


def load(name, m, seed):
    net = fusion.FusionNet(m, residual=True).to(device)
    net.load_state_dict(torch.load(f"runs/tuning/final/{name}_s{seed}.pth", map_location=device)["model"])
    return net


def main():
    base, radar = cache_cells.load("val", "runs/cells", *GATE)
    second = scene_scores(None, base, radar, [])
    tuned = [scene_scores(load("T2", 1, s), base, radar, [1]) for s in (1, 2, 3)]
    no_radar = [scene_scores(load("R0+T2", 0, s), base, radar, []) for s in (1, 2, 3)]
    scenes = [n for n, v in second.items() if v["n_gt"] >= MIN_CARS]
    table = {n: dict(n_gt=second[n]["n_gt"], second=second[n]["APCD"], R0_T2=float(np.mean([r[n]["APCD"] for r in no_radar])), R2_T2=float(np.mean([r[n]["APCD"] for r in tuned]))) for n in scenes}
    d_radar = np.array([v["R2_T2"] - v["R0_T2"] for v in table.values()])
    d_second = np.array([v["R2_T2"] - v["second"] for v in table.values()])
    counts = dict(scenes=len(scenes), R2_T2_above_R0_T2=int((d_radar > 0).sum()), R2_T2_above_second=int((d_second > 0).sum()), median_vs_R0_T2=float(np.median(d_radar)), median_vs_second=float(np.median(d_second)))
    print(counts)
    with open(OUT, "w") as f:
        json.dump(dict(counts=counts, scenes=table), f, indent=1)


if __name__ == "__main__":
    main()
