"""Why does radar add so little at range: is the information missing, or does the fusion network fail to extract it?

1. Training examples by distance, and radar's separation on training frames (a quarter of them).
2. An information test: logistic regression on SECOND's score, with and without radar, scored on scenes it did not train on.
3. The paper's selection metric on val, per seed and per scene, for SECOND, R0 and R3.
Analysis only. Nothing here feeds back into the model or the gate.

Usage, from the repo root: python results/clocs_radar/diag_radar_limits.py [--parts train,information,far]
"""
import argparse
import json
import pathlib

import numpy as np
import torch
from clocs_r import aimotive, cache_cells, fusion, metrics, radar_features, train_fusion

OUT = pathlib.Path(__file__).parent
SEEDS = (1, 2, 3)
GATE = (0.5, 0.02, 3)
R3 = [radar_features.FEATURES.index(f) for f in radar_features.FEATURE_SETS["R3"]]
FAR = (50, 200)
device = torch.device("cuda")


def kept_boxes(base, radar):
    """Boxes SECOND's NMS keeps: distance, real car, SECOND score, support, Doppler speed, scene index."""
    rows, names = [], {}
    for it, feats in zip(base, radar):
        mask = np.nonzero(it["keep_eval"])[0]
        kept = mask[metrics.nms_bev(it["boxes"][mask], it["scores"][mask], 0.1)]
        real = metrics.iou_bev(it["gt"], it["boxes"][kept]).max(0) >= 0.3 if len(it["gt"]) and len(kept) else np.zeros(len(kept), bool)
        feats = np.asarray(feats[kept], np.float32)
        scene = names.setdefault(it["seq"], len(names))
        rows.append(np.c_[it["dist"][kept] * cache_cells.MAX_RANGE, real, it["scores"][kept], feats[:, 0], feats[:, 1], np.full(len(kept), scene)])
    return np.concatenate(rows)


def in_bin(a, lo, hi):
    return a[(a[:, 0] >= lo) & (a[:, 0] < hi)]


def auc(score, label):
    pos, neg = int(label.sum()), int((~label).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    # Average ranks, so tied scores count half
    values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def features(s, radar):
    """SECOND's logit and its square, plus, with radar, whether there is a return, the support and Doppler speed, and how they interact with the score."""
    p = np.clip(s[:, 2], 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    x = [z, z ** 2]
    if radar:
        hit = (s[:, 3] > 0).astype(float)
        x += [hit, s[:, 3], hit * s[:, 4], hit * z, hit * s[:, 4] * z]
    return np.stack(x, 1)


def fit_logistic(x, y, ridge=1e-3, steps=25):
    """Newton's method with a small ridge. Returns the mean, scale and weights."""
    mean, scale = x.mean(0), x.std(0) + 1e-9
    a = np.c_[np.ones(len(x)), (x - mean) / scale]
    w = np.zeros(a.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(a @ w, -30, 30)))
        h = (a * (p * (1 - p) + 1e-9)[:, None]).T @ a / len(a) + ridge * np.eye(a.shape[1])
        step = np.linalg.solve(h, (a.T @ (p - y)) / len(a) + ridge * w)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return mean, scale, w


def predict_logistic(model, x):
    mean, scale, w = model
    return np.c_[np.ones(len(x)), (x - mean) / scale] @ w


def fit_predict(train, test, radar):
    return predict_logistic(fit_logistic(features(train, radar), (train[:, 1] > 0).astype(float)), features(test, radar))


def cross_fit(s, folds=5):
    """AUC on held-out scenes, without and with radar. Folds never split a scene."""
    y, scene = s[:, 1] > 0, s[:, 5].astype(int)
    p0, p1 = np.zeros(len(s)), np.zeros(len(s))
    for k in range(folds):
        test = scene % folds == k
        p0[test], p1[test] = fit_predict(s[~test], s[test], False), fit_predict(s[~test], s[test], True)
    return dict(boxes=len(s), real=int(y.sum()), auc_second_alone=auc(s[:, 2], y), auc_without_radar=auc(p0, y), auc_with_radar=auc(p1, y))


def load_net(path, m):
    net = fusion.FusionNet(m).to(device)
    net.load_state_dict(torch.load(path, map_location=device)["model"])
    return net


def far_metric(net, base, radar, idx):
    """Selection metric overall and per scene, with detections kept once so each subset reuses them."""
    records = [(it["gt"], b, s, it["seq"]) for it, b, s in train_fusion.predict(net, base, radar, idx, device)]
    name = f"{FAR[0]}-{FAR[1]}"
    far = metrics.evaluate([r[:3] for r in records], bins=[FAR])[name]
    scenes = {}
    for scene in sorted({r[3] for r in records}):
        v = metrics.evaluate([r[:3] for r in records if r[3] == scene], bins=[FAR])[name]
        scenes[scene] = dict(n_gt=v["n_gt"], APCD=v["APCD"], APBEV=v["APBEV@0.3"])
    return dict(APCD=far["APCD"], APBEV=far["APBEV@0.3"], n_gt=far["n_gt"]), scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="train,information,far")
    parts = ap.parse_args().parts.split(",")
    path = OUT / "diag_radar_limits.json"
    results = json.load(open(path)) if path.exists() else {}

    if "train" in parts:
        base, radar = cache_cells.load("train", "runs/cells", *GATE)
        results["train_examples"] = {}
        for lo, hi in aimotive.DIST_BINS:
            pos, neg = 0, 0
            for it in base:
                rho = it["dist"] * cache_cells.MAX_RANGE
                m = (rho >= lo) & (rho < hi) & (it["weight"] > 0)
                pos += int(it["target"][m].sum())
                neg += int(m.sum() - it["target"][m].sum())
            results["train_examples"][f"{lo}-{hi}"] = dict(positives=pos, negatives=neg)
        a = kept_boxes(base[::4], radar[::4])
        results["train_signal"] = {}
        for lo, hi in aimotive.DIST_BINS:
            s = in_bin(a, lo, hi)
            y = s[:, 1] > 0
            results["train_signal"][f"{lo}-{hi}"] = dict(real=int(y.sum()), real_with_return=float((s[y, 3] > 0).mean()), false_with_return=float((s[~y, 3] > 0).mean()),
                                                         auc_second=auc(s[:, 2], y), auc_support=auc(s[:, 3], y), auc_doppler=auc(s[:, 4], y))
        del base, radar, a

    if "information" in parts:
        data = {}
        for split in ("holdout", "val"):
            base, radar = cache_cells.load(split, "runs/cells", *GATE)
            data[split] = kept_boxes(base, radar)
            results[f"information_{split}"] = {f"{lo}-{hi}": cross_fit(in_bin(data[split], lo, hi)) for lo, hi in aimotive.DIST_BINS}
            print(split, json.dumps(results[f"information_{split}"]), flush=True)
        results["information_holdout_to_val"] = {}
        for lo, hi in aimotive.DIST_BINS:
            h, v = in_bin(data["holdout"], lo, hi), in_bin(data["val"], lo, hi)
            results["information_holdout_to_val"][f"{lo}-{hi}"] = dict(auc_without_radar=auc(fit_predict(h, v, False), v[:, 1] > 0), auc_with_radar=auc(fit_predict(h, v, True), v[:, 1] > 0))
        print("holdout->val", json.dumps(results["information_holdout_to_val"]), flush=True)

    if "far" in parts:
        base, radar = cache_cells.load("val", "runs/cells", *GATE)
        second, second_scenes = far_metric(None, base, radar, [])
        r0 = [far_metric(load_net(f"runs/fusion/R0_s{s}/fusion.pth", 0), base, radar, []) for s in SEEDS]
        r3 = [far_metric(load_net(f"runs/fusion/R3_s{s}/fusion.pth", len(R3)), base, radar, R3) for s in SEEDS]
        results["far_val"] = dict(second=second, R0=[x[0] for x in r0], R3=[x[0] for x in r3])
        # Scenes with at least 50 far cars, seed means. Paired counts, because the gains are small next to the seed spread.
        scenes = [n for n, v in second_scenes.items() if v["n_gt"] >= 50]
        per = lambda runs, n, m: float(np.mean([x[1][n][m] for x in runs]))
        results["far_val_scenes"] = {n: dict(n_gt=second_scenes[n]["n_gt"], second=second_scenes[n]["APCD"], R0=per(r0, n, "APCD"), R3=per(r3, n, "APCD")) for n in scenes}
        d0 = np.array([v["R3"] - v["R0"] for v in results["far_val_scenes"].values()])
        d1 = np.array([v["R3"] - v["second"] for v in results["far_val_scenes"].values()])
        results["far_val_counts"] = dict(scenes=len(scenes), R3_above_R0=int((d0 > 0).sum()), R3_above_second=int((d1 > 0).sum()), median_vs_R0=float(np.median(d0)), median_vs_second=float(np.median(d1)))
        print(json.dumps({k: results[k] for k in ("far_val", "far_val_counts")}), flush=True)

    with open(path, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
