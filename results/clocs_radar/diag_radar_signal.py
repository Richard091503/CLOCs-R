"""How well does each score separate real cars from false detections, by distance, and what did the network learn from radar?

Same definitions as ../clocs_no_radar/diag_camera_signal.py: the boxes are those SECOND's own NMS keeps, and a box is a real car at
BEV IoU >= 0.3 with a label. Each of the three trained R3 networks is also run with its radar inputs replaced by "no return".

Usage, from the repo root: python results/clocs_radar/diag_radar_signal.py
"""
import json
import pathlib

import numpy as np
import torch

from clocs_r import aimotive, cache_cells, fusion, metrics, radar_features, train_fusion

OUT = pathlib.Path(__file__).parent
SEEDS = (1, 2, 3)
GATE = (0.5, 0.02, 3)
R3 = [radar_features.FEATURES.index(f) for f in radar_features.FEATURE_SETS["R3"]]
device = torch.device("cuda")


def auc(score, label):
    pos, neg = int(label.sum()), int((~label).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    # Average ranks, so tied scores count half
    values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def load_net(path, m):
    net = fusion.FusionNet(m).to(device)
    net.load_state_dict(torch.load(path, map_location=device)["model"])
    return net.eval()


@torch.no_grad()
def logits(net, it, radar, names_idx):
    x, col = train_fusion.frame_cells(it, radar, names_idx, device)
    return net(x, col, len(it["scores"])).cpu().numpy()


def collect(base, radar, r0, r3):
    rows = []
    missing = np.zeros((radar.shape[1], radar.shape[2]), np.float32) - 1
    missing[:, 0] = 0
    for it, feats in zip(base, radar):
        mask = np.nonzero(it["keep_eval"])[0]
        kept = mask[metrics.nms_bev(it["boxes"][mask], it["scores"][mask], 0.1)]
        real = metrics.iou_bev(it["gt"], it["boxes"][kept]).max(0) >= 0.3 if len(it["gt"]) and len(kept) else np.zeros(len(kept), bool)
        feats = np.asarray(feats, np.float32)
        s_r0 = np.mean([logits(n, it, None, [])[kept] for n in r0], 0)
        with_radar = [logits(n, it, feats, R3)[kept] for n in r3]
        without = [logits(n, it, missing, R3)[kept] for n in r3]
        rows.append(np.c_[it["dist"][kept] * cache_cells.MAX_RANGE, real, it["scores"][kept], feats[kept, 0], feats[kept, 1], s_r0,
                          np.mean(with_radar, 0), np.mean(without, 0)])
    return np.concatenate(rows)


def main():
    r0 = [load_net(f"runs/fusion/R0_s{s}/fusion.pth", 0) for s in SEEDS]
    r3 = [load_net(f"runs/fusion/R3_s{s}/fusion.pth", len(R3)) for s in SEEDS]
    results = {}
    for split in ("holdout", "val"):
        base, radar = cache_cells.load(split, "runs/cells", *GATE)
        a = collect(base, radar, r0, r3)
        results[split] = {}
        for lo, hi in aimotive.DIST_BINS:
            s = a[(a[:, 0] >= lo) & (a[:, 0] < hi)]
            real = s[:, 1] > 0
            change = s[:, 6] - s[:, 7]  # logit with radar minus logit with the radar inputs marked missing
            hit = s[:, 3] > 0
            r = dict(boxes=len(s), real_cars=int(real.sum()), real_with_return=float(hit[real].mean()), false_with_return=float(hit[~real].mean()),
                     auc_second=auc(s[:, 2], real), auc_support=auc(s[:, 3], real), auc_doppler=auc(s[:, 4], real), auc_r0=auc(s[:, 5], real), auc_r3=auc(s[:, 6], real),
                     auc_r3_radar_removed=auc(s[:, 7], real), logit_change_real_with_return=float(change[real & hit].mean()), logit_change_false_with_return=float(change[~real & hit].mean()),
                     logit_change_real_no_return=float(change[real & ~hit].mean()), logit_change_false_no_return=float(change[~real & ~hit].mean()))
            results[split][f"{lo}-{hi}"] = r
            print(split, f"{lo}-{hi}", {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    with open(OUT / "diag_radar_signal.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
