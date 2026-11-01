"""How well does each score separate real cars from false detections, by distance? One-off check, from the repo root.

The boxes are those SECOND's own NMS keeps, so only the ranking differs. A box counts as a real car at BEV IoU >= 0.3 with a label.
AUC 0.5 means no signal and 1.0 means perfect separation.

Usage: python results/clocs_no_radar/diag_camera_signal.py
"""
import json
import pathlib

import numpy as np
import torch

from clocs_r import aimotive, cache_cells, fusion, metrics, train_fusion

OUT = pathlib.Path(__file__).parent
device = torch.device("cuda")


def auc(score, label):
    pos, neg = int(label.sum()), int((~label).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    # Average ranks, so tied scores count half
    values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def collect(base, net):
    rows = []
    for it in base:
        x, col = train_fusion.frame_cells(it, None, [], device)
        with torch.no_grad():
            fused = torch.sigmoid(net(x, col, len(it["scores"]))).cpu().numpy()
        mask = np.nonzero(it["keep_eval"])[0]
        kept = mask[metrics.nms_bev(it["boxes"][mask], it["scores"][mask], 0.1)]
        any_2d = np.zeros(len(it["scores"]), np.float32)
        aligned_2d = np.zeros(len(it["scores"]), np.float32)
        iou, s2 = it["pairs"][:, 0], it["pairs"][:, 1]
        np.maximum.at(any_2d, it["column"], np.maximum(s2, 0))
        good = iou >= 0.5
        np.maximum.at(aligned_2d, it["column"][good], s2[good])
        real = metrics.iou_bev(it["gt"], it["boxes"][kept]).max(0) >= 0.3 if len(it["gt"]) else np.zeros(len(kept), bool)
        rows.append(np.c_[it["dist"][kept] * cache_cells.MAX_RANGE, it["scores"][kept], any_2d[kept], aligned_2d[kept], fused[kept], real])
    return np.concatenate(rows)


def main():
    net = fusion.FusionNet(0).to(device)
    net.load_state_dict(torch.load("runs/fusion/R0_s3/fusion.pth", map_location=device)["model"])
    net.eval()
    results = {}
    for split in ("holdout", "val"):
        base, _ = cache_cells.load(split, "runs/cells", 0.5, 0.02, 3, use_radar=False)
        a = collect(base, net)
        results[split] = {}
        for lo, hi in aimotive.DIST_BINS:
            s = a[(a[:, 0] >= lo) & (a[:, 0] < hi)]
            real = s[:, 5] > 0
            r = dict(boxes=len(s), real_cars=int(real.sum()), auc_second=auc(s[:, 1], real), auc_2d_any_overlap=auc(s[:, 2], real),
                     auc_2d_aligned=auc(s[:, 3], real), auc_clocs=auc(s[:, 4], real),
                     real_with_aligned_2d_above_0p5=float((s[real, 3] >= 0.5).mean()) if real.any() else float("nan"))
            results[split][f"{lo}-{hi}"] = r
            print(split, f"{lo}-{hi}", {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    with open(OUT / "diag_camera_signal.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
