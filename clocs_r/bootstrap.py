"""Scene-level bootstrap intervals for APCD differences (results/next_round/REEVALUATION.md section 4, PLAN.md Phase 7).

Each model is scored per frame (kept boxes after NMS), every detection's centre-distance match is recorded per distance
bin and threshold, and scenes are resampled with replacement. A resample weights each frame by how many times its scene
was drawn, so AP is computed once per resample from pre-sorted detections (weighted cumulative sums), exactly equal to
metrics.evaluate's APCD when every weight is 1. Seeds are averaged inside each resample, and paired differences between
models get percentile intervals.

    python -m clocs_r.bootstrap --split select --out results/next_round/phase7/bootstrap_select.json
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from clocs_r import metrics, selector
from clocs_r import tune_fusion as tf

BINS = {"50-100": (50, 100), "100-200": (100, 200), "50-200": (50, 200), "all": (0, float("inf"))}
P1 = "runs/next_round/phase1"


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


@torch.no_grad()
def fusion_scores(name, weights, seed, d, device, key="bs"):
    """Sigmoid scores of a saved tune_fusion network on the split held in d (selector.load_split format)."""
    cfg = tf.parse(name)
    idx = tf.names_idx(cfg)
    items = d["rc_items"] if cfg.get("rc") else d["rp_items"] if cfg.get("rp") else d["items"]
    tf.DATA.update({key: items, "radar_" + key: d["radar"], "vis_" + key: d["vis"], "moving_" + key: d["moving"]})
    if cfg.get("gated"):
        tf.DATA["ctx_" + key] = tf.gate_context(cfg, key, items, d["radar"])
    extra, pairing = tf.p3_inputs(cfg, key)
    _, net = selector.load_expert(name, weights, seed, device)
    return [1 / (1 + np.exp(-np.clip(z, -30, 30))) for z in selector.expert_logits(net, cfg, items, d["radar"], extra, pairing, device)]


@torch.no_grad()
def selector_scores(radar_expert, radar_weights, gate_dir, seed, d, device, kind="selector"):
    gate = selector.load_gate(gate_dir, radar_expert, seed, device, kind)
    lg = selector.test_logits(d, radar_expert, radar_weights, seed, device)
    ctx = selector.frame_contexts(d, lg)
    return [torch.sigmoid(gate(torch.from_numpy(c).to(device), torch.from_numpy(z).to(device))).cpu().numpy() for c, z in zip(ctx, lg)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--models", required=True, help="json list of [label, kind, name, weights] (kind: second, fusion, selector, blend)")
    ap.add_argument("--comparisons", required=True, help="json list of [label A, label B]")
    ap.add_argument("--radar_dets", default="", help="radar-detector boxes for RP models and the selector's RP expert")
    ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="first N frames only (tests)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    models = json.loads(args.models)
    seeds = [int(s) for s in args.seeds.split(",")]
    d = selector.load_split(args.split, f"{P1}/cells_in", f"{P1}/cand_in/3d", args.radar_dets or None)
    if args.limit:
        d = selector.subset(d, np.arange(args.limit))
    items = d["items"]
    matches = {}
    for label, kind, name, weights in models:
        for seed in ([0] if kind == "second" else seeds):
            if kind == "second":
                sc = [it["scores"] for it in items]
            elif kind == "fusion":
                sc = fusion_scores(name, weights, seed, d, device)
            else:
                radar_expert, radar_weights = name.split("|")
                sc = selector_scores(radar_expert, radar_weights, weights, seed, d, device, kind)
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
