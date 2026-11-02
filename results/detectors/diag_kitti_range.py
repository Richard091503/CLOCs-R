"""aiMotive val scored over KITTI's range, so the overall scores of the two datasets can be compared. SECOND and CLOCs without radar.

Usage, from the repo root: python results/detectors/diag_kitti_range.py
"""
import json
import pathlib

import torch

from clocs_r import cache_cells, fusion, metrics, train_fusion

OUT = pathlib.Path(__file__).parent / "diag_kitti_range.json"
BINS = [(0, 70.4), (50, 70.4)]
KEEP = ("n_gt", "APBEV@0.3", "APCD", "AP3D@0.5", "AP3D@0.7")


def score(net, base, device):
    records = [(it["gt"], b, s) for it, b, s in train_fusion.predict(net, base, [None] * len(base), [], device)]
    return {k: {m: v[m] for m in KEEP} for k, v in metrics.evaluate(records, bins=BINS).items()}


def main():
    device = torch.device("cuda")
    base, _ = cache_cells.load("val", "runs/cells", 0.5, 0.02, 3, use_radar=False)
    results = dict(second=score(None, base, device), R0=[])
    for seed in (1, 2, 3):
        net = fusion.FusionNet(0).to(device)
        net.load_state_dict(torch.load(f"runs/fusion/R0_s{seed}/fusion.pth", map_location=device)["model"])
        results["R0"].append(score(net, base, device))
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
