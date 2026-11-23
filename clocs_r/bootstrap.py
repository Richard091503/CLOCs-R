"""Scene-level bootstrap intervals for APCD differences (docs/MODELS.md; the method is set out in the
thesis record's REEVALUATION.md, section 4).

Each model is scored per frame (kept boxes after NMS), every detection's centre-distance match is recorded per distance
bin and threshold, and scenes are resampled with replacement. A resample weights each frame by how many times its scene
was drawn, so AP is computed once per resample from pre-sorted detections (weighted cumulative sums), exactly equal to
metrics.evaluate's APCD when every weight is 1. Seeds are averaged inside each resample, and paired differences between
models get percentile intervals.

    python -m clocs_r.bootstrap --split val --models '[["L","second","",""],["LCR","fusion","R0+T2+K40+RC","runs/lcr/fusion"]]' \
        --comparisons '[["LCR","L"]]' --out results/lcr/bootstrap_val.json
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from clocs_r import metrics, radar_pairs, train_fusion
from clocs_r import tune_fusion as tf

BINS = {"50-100": (50, 100), "100-200": (100, 200), "50-200": (50, 200), "all": (0, float("inf"))}


def frame_matches(frames):
    """Per bin: (scores, tp (D, T) bool, frame index (D,), n_gt per frame (F,)) over all frames, sorted by score."""
    out = {}
    for name, (lo, hi) in BINS.items():
        sc, tp, fi, ngt = [], [], [], np.zeros(len(frames), np.int64)
        for f, (gt, det, score) in enumerate(frames):
            order = np.argsort(-score)
            det, score = det[order], score[order]
            rg, rd = np.hypot(gt[:, 0], gt[:, 1]), np.hypot(det[:, 0], det[:, 1])
            g, d, s = gt[(rg >= lo) & (rg < hi)], det[(rd >= lo) & (rd < hi)], score[(rd >= lo) & (rd < hi)]
            ngt[f] = len(g)
            dc = np.hypot(g[:, None, 0] - d[None, :, 0], g[:, None, 1] - d[None, :, 1]) if len(g) and len(d) else np.zeros((len(g), len(d)))
            tp.append(np.stack([metrics._match(g, d, dc, False, t)[0] for t in metrics.CD_THRESHOLDS], 1) if len(d)
                      else np.zeros((0, len(metrics.CD_THRESHOLDS)), bool))
            sc.append(s)
            fi.append(np.full(len(s), f))
        sc, tp, fi = np.concatenate(sc), np.concatenate(tp), np.concatenate(fi)
        order = np.argsort(-sc, kind="mergesort")  # the same stable order metrics.evaluate uses within its concatenation
        out[name] = (sc[order], tp[order], fi[order], ngt)
    return out


def weighted_apcd(m, w):
    """APCD of one bin under frame weights w (F,): metrics._ap with each detection counted w[frame] times."""
    sc, tp, fi, ngt = m
    n_gt = float((w * ngt).sum())
    if n_gt == 0:
        return float("nan")
    wd = w[fi].astype(np.float64)
    keep = wd > 0
    wd, tpk = wd[keep], tp[keep]
    if len(wd) == 0:
        return 0.0
    cum_w = np.cumsum(wd)
    aps = []
    for t in range(tpk.shape[1]):
        tps = np.cumsum(wd * tpk[:, t])
        recall = np.concatenate([[0.0], tps / n_gt, [1.0]])
        precision = np.concatenate([[1.0], tps / cum_w, [0.0]])
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        aps.append(float(np.sum((recall[1:] - recall[:-1]) * precision[1:])))
    return float(np.mean(aps))


def kept_frames(items, scores):
    out = []
    for it, s in zip(items, scores):
        m = it["keep_eval"]
        b, ss = it["boxes"][m], np.asarray(s, np.float32)[m]
        keep = metrics.nms_bev(b, ss, 0.1)
        out.append((it["gt"], b[keep], ss[keep]))
    return out


def load_split(split, cells, cand3d, cand2d, radar_dets):
    """A split's cached frames, in the three pairings the models use: base (L, LC), rp (LRd) and rc (LCR, LCR-D)."""
    tf.prepare(split, "bs", cells, cand3d, cand2d, None, "base")
    items, top2d = tf.DATA.pop("bs"), tf.DATA.pop("top2d_bs")
    rp = radar_pairs.build(split, cells, radar_dets, items)
    return dict(items=items, top2d=top2d, rp=radar_pairs.replace_pairs(items, rp), rc=radar_pairs.add_pairs(items, rp, top2d))


def subset(d, idx):
    return dict(items=[d["items"][i] for i in idx], top2d=d["top2d"][idx], rp=[d["rp"][i] for i in idx], rc=[d["rc"][i] for i in idx])


@torch.no_grad()
def fusion_scores(name, weights, seed, d, device):
    """Sigmoid scores of a saved tune_fusion network on the split held in d (load_split format)."""
    cfg = tf.parse(name)
    items = d[tf.kind(cfg)] if tf.kind(cfg) != "base" else d["items"]
    net = tf.load_net(cfg, name, weights, seed, 2.0, device).eval()
    pairing = tf.pairing(cfg, d["top2d"])
    out = []
    for i, item in enumerate(items):
        x, col = train_fusion.frame_cells(item, device, pairing[i] if pairing is not None else None)
        z = net(x, col, len(item["scores"])).float().cpu().numpy()
        out.append(1 / (1 + np.exp(-np.clip(z, -30, 30))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--models", required=True, help="json list of [label, kind, name, weights] (kind: second or fusion)")
    ap.add_argument("--comparisons", required=True, help="json list of [label A, label B]")
    ap.add_argument("--radar_dets", default="runs/lcr/d2_dets", help="exported D2 boxes, for RP and RC models")
    ap.add_argument("--cells", default="runs/lcr/cells")
    ap.add_argument("--cand3d", default="runs/lcr/cand/3d")
    ap.add_argument("--cand2d", default="runs/lcr/cand/2d")
    ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="first N frames only (tests)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    models = json.loads(args.models)
    seeds = [int(s) for s in args.seeds.split(",")]
    d = load_split(args.split, args.cells, args.cand3d, args.cand2d, args.radar_dets)
    if args.limit:
        d = subset(d, np.arange(args.limit))
    items = d["items"]
    matches = {}
    for label, kind, name, weights in models:
        for seed in ([0] if kind == "second" else seeds):
            if kind == "second":
                sc = [it["scores"] for it in items]
            else:
                assert kind == "fusion", kind
                sc = fusion_scores(name, weights, seed, d, device)
            matches[(label, seed)] = frame_matches(kept_frames(items, sc))
        print("scored", label, flush=True)
    scenes = sorted({it["seq"] for it in items})
    frame_scene = np.array([scenes.index(it["seq"]) for it in items])
    rng = np.random.RandomState(0)

    def value(label, w, b):
        ks = [k for k in matches if k[0] == label]
        return float(np.mean([weighted_apcd(matches[k][b], w) for k in ks]))

    ones = np.ones(len(items))
    point = {label: {b: value(label, ones, b) for b in BINS} for label, *_ in models}
    draws = {tuple(c): {b: [] for b in BINS} for c in json.loads(args.comparisons)}
    for _ in range(args.resamples):
        counts = np.bincount(rng.randint(0, len(scenes), len(scenes)), minlength=len(scenes))
        w = counts[frame_scene].astype(np.float64)
        cache = {}
        for (a, bb) in draws:
            for b in BINS:
                for lab in (a, bb):
                    if (lab, b) not in cache:
                        cache[(lab, b)] = value(lab, w, b)
                draws[(a, bb)][b].append(cache[(a, b)] - cache[(bb, b)])
    out = dict(split=args.split, scenes=len(scenes), frames=len(items), resamples=args.resamples, point=point, comparisons={})
    for (a, bb), per in draws.items():
        out["comparisons"][f"{a} - {bb}"] = {b: dict(diff=point[a][b] - point[bb][b], lo=float(np.percentile(v, 2.5)),
                                                     hi=float(np.percentile(v, 97.5)), p_le_0=float(np.mean(np.array(v) <= 0)))
                                            for b, v in per.items()}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    for k, v in out["comparisons"].items():
        print(k, {b: f"{x['diff']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]" for b, x in v.items()}, flush=True)


if __name__ == "__main__":
    main()
