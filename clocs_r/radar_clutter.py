"""Per-point clutter classifier for radar (results/next_round/PLAN.md, Phase 6b, detector D2).

Input per radar point: its six radar_dataset features, its range, and statistics of its BEV neighbourhood (within
RADIUS m, over all accumulated sweeps): log(1 + returns), distinct sweeps / sweeps, mean and SD of Doppler, max RCS.
Label: 1 if the point lies (in BEV) inside a car box enlarged by MARGIN m on every side, else 0. A small MLP gives the car
probability, used by D2 as a seventh point feature.

    python -m clocs_r.radar_clutter --out runs/next_round/phase6b/clutter.pt --report results/next_round/phase6b/clutter.json
"""
import argparse
import json
import multiprocessing as mp
import pathlib

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from clocs_r import cache_cells, radar_dataset, splits

RADIUS, MARGIN = 2.0, 0.5
N_FEATURES = 12


def features(points, sweeps=radar_dataset.SWEEPS):
    """(N, 12) float32 classifier input of radar_dataset points (N, >=6)."""
    n = len(points)
    out = np.zeros((n, N_FEATURES), np.float32)
    if n == 0:
        return out
    out[:, :6] = points[:, :6]
    out[:, 6] = np.hypot(points[:, 0], points[:, 1]) / 200.0
    near = cKDTree(points[:, :2]).query_ball_point(points[:, :2], RADIUS)
    age = np.round(points[:, 5] / max(points[:, 5].max(), 1e-6) * (sweeps - 1)) if points[:, 5].max() > 0 else np.zeros(n)
    for i, nb in enumerate(near):
        nb = np.asarray(nb)
        v = points[nb, 3]
        out[i, 7] = np.log1p(len(nb))
        out[i, 8] = len(np.unique(age[nb])) / sweeps
        out[i, 9] = v.mean()
        out[i, 10] = v.std()
        out[i, 11] = points[nb, 4].max()
    return out


def labels(seq, frame, points):
    """1 for points inside (BEV) any car box enlarged by MARGIN on every side."""
    boxes, _ = seq.boxes(frame, splits.CAR_CLASSES)
    lab = np.zeros(len(points), np.float32)
    if len(boxes) == 0 or len(points) == 0:
        return lab
    dx = points[None, :, 0] - boxes[:, None, 0]
    dy = points[None, :, 1] - boxes[:, None, 1]
    c, s = np.cos(boxes[:, 6])[:, None], np.sin(boxes[:, 6])[:, None]
    inside = (np.abs(dx * c + dy * s) <= boxes[:, None, 3] / 2 + MARGIN) & (np.abs(-dx * s + dy * c) <= boxes[:, None, 4] / 2 + MARGIN)
    return inside.any(0).astype(np.float32)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEATURES, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)[:, 0]


def _work(job):
    path, frame, motion = job
    seq = cache_cells._seq(path)
    pts = radar_dataset.radar_points(seq, frame, motion=motion)
    return features(pts), labels(seq, frame, pts)


def dataset(split, motion=True, workers=8):
    with mp.Pool(workers) as pool:
        out = pool.map(_work, [(s.path, f, motion) for s, f in splits.frames(split)], chunksize=32)
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def auc(score, y):
    """Area under the ROC curve by the rank-sum statistic (ties get average ranks)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def train(x, y, epochs=5, seed=0, batch=4096):
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0) + 1e-6
    xt = torch.from_numpy((x - mean) / std)
    yt = torch.from_numpy(y)
    net = MLP()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(xt), generator=g).split(batch):
            opt.zero_grad()
            loss = lossf(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return dict(model=net.state_dict(), mean=mean, std=std)


def load(path):
    state = torch.load(path, map_location="cpu")
    net = MLP()
    net.load_state_dict(state["model"])
    net.eval()
    return net, state["mean"], state["std"]


@torch.no_grad()
def predict(clf, points):
    net, mean, std = clf
    if len(points) == 0:
        return np.zeros(0, np.float32)
    x = (features(points) - mean) / std
    return torch.sigmoid(net(torch.from_numpy(x.astype(np.float32)))).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--no_motion", action="store_true")
    args = ap.parse_args()
    motion = not args.no_motion
    torch.set_num_threads(8)
    x, y = dataset("fit", motion)
    state = train(x, y)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state, motion=motion), args.out)
    clf = load(args.out)
    rep = dict(fit_points=int(len(y)), fit_positive=float(y.mean()), motion=motion)
    for split in ("fit", "select"):
        xs, ys = (x, y) if split == "fit" else dataset(split, motion)
        with torch.no_grad():
            p = torch.sigmoid(clf[0](torch.from_numpy(((xs - clf[1]) / clf[2]).astype(np.float32)))).numpy()
        rep[f"auc_{split}"] = auc(p, ys)
        rep[f"positive_{split}"] = float(ys.mean())
        rep[f"auc_{split}_far"] = auc(p[xs[:, 6] * 200 >= 50], ys[xs[:, 6] * 200 >= 50])
    pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(args.report, "w"), indent=1)
    print(rep, flush=True)


if __name__ == "__main__":
    main()
