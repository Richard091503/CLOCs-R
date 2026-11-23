"""Phase 6 extras (results/next_round/PLAN.md): SECOND ∪ radar, recall gained and false alarms added, on the frames
with the radar-sourced candidates appended. Variants: "none" drops the radar-sourced candidates, "union" keeps them.

    python results/next_round/phase6_extra.py --out results/next_round/phase6/extra.json
"""
import argparse
import json
import multiprocessing as mp
import pathlib

import numpy as np
import torch

from clocs_r import aimotive, cache_cells, metrics, radar_candidates, splits, train_fusion, visibility
from clocs_r import oracle_inject as oi
from clocs_r import tune_fusion as tf

P1 = "runs/next_round/phase1"
MODELS = {"SECOND": None, "R0+T2+K40+G": "runs/next_round/phase4/fusion", "R2+T2+K40+G": "runs/next_round/phase4/fusion",
          "R2+T2+K40+G+RD": "runs/next_round/phase6/fusion"}
VARIANTS = {"SECOND": ("none", "union"), "R0+T2+K40+G": ("none",), "R2+T2+K40+G": ("none",), "R2+T2+K40+G+RD": ("none", "union")}
MIN_SCORE, CD = 0.1, 2.0
FAR_BINS = ((50, 100), (100, 200), (50, 200))


def frames_after_nms(base, scores, variant):
    """Per frame: gt, kept boxes, kept scores, kept src flags, odd, seq."""
    out = []
    for item, s in zip(base, scores):
        s = np.asarray(s, np.float32)
        mask = item["keep_eval"] & (~item["src"] if variant == "none" else True)
        b, ss, src = item["boxes"][mask], s[mask], item["src"][mask]
        keep = metrics.nms_bev(b, ss, 0.1)
        out.append((item["gt"], b[keep], ss[keep], src[keep], item["odd"], item["seq"]))
    return out


def recall_and_false_alarms(frames):
    """By condition: recall of scored cars beyond 50 m (a kept box with score >= MIN_SCORE within CD), and radar-sourced
    kept boxes with score >= MIN_SCORE: count, true (within CD of a scored car) and false alarms per frame."""
    out = {}
    groups = train_fusion.condition_groups(frames, odd=lambda r: r[4], seq=lambda r: r[5])
    for name, rows in groups.items():
        found = {f"{lo}-{hi}": [0, 0] for lo, hi in FAR_BINS}
        radar_kept = radar_true = 0
        for gt, b, s, src, _, _ in rows:
            strong = s >= MIN_SCORE
            if len(gt):
                rng = np.hypot(gt[:, 0], gt[:, 1])
                hit = radar_candidates.nearest(gt, b[strong]) <= CD
                for lo, hi in FAR_BINS:
                    m = (rng >= lo) & (rng < hi)
                    found[f"{lo}-{hi}"][0] += int((hit & m).sum())
                    found[f"{lo}-{hi}"][1] += int(m.sum())
            r = strong & src
            radar_kept += int(r.sum())
            radar_true += int((radar_candidates.nearest(b[r], gt) <= CD).sum()) if r.any() else 0
        out[name] = dict(recall={k: v[0] / max(v[1], 1) for k, v in found.items()}, cars={k: v[1] for k, v in found.items()},
                         radar_kept=radar_kept, radar_true=radar_true, false_alarms_per_frame=(radar_kept - radar_true) / len(rows),
                         frames=len(rows))
    return out


def score(base, scores, variants):
    res = {}
    for v in variants:
        fr = frames_after_nms(base, scores, v)
        res[v] = {name: metrics.evaluate([r[:3] for r in rows], bins=aimotive.DIST_BINS + [train_fusion.FAR])
                  for name, rows in train_fusion.condition_groups(fr, odd=lambda r: r[4], seq=lambda r: r[5]).items()}
        res[v]["recall_fa"] = recall_and_false_alarms(fr)
    return res


def run_seed(job):
    name, seed, eval_splits = job
    torch.set_num_threads(1)
    device = torch.device("cuda")
    cfg = tf.parse(name)
    idx = tf.names_idx(cfg)
    path = pathlib.Path(MODELS[name]) / f"{name}_s{seed}.pth"
    state = torch.load(path, map_location="cpu")
    assert state["settings"] == cfg and state.get("far_weight") == 2.0, path
    net = tf.make_net(cfg, idx).to(device)
    net.load_state_dict(state["model"])
    net.eval()
    return dict(seed=seed, **{s: score(tf.DATA[s], oi.net_scores(net, cfg, s, device), VARIANTS[name]) for s in eval_splits})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="select,val")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--cells", default=f"{P1}/cells_in")
    ap.add_argument("--cand3d", default=f"{P1}/cand_in/3d")
    ap.add_argument("--cand2d", default=f"{P1}/cand_in/2d")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    eval_splits = args.splits.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    out = pathlib.Path(args.out)
    results = json.load(open(out)) if out.exists() else {}
    for s in eval_splits:
        base, _ = cache_cells.load(s, args.cells, *tf.GATE, cache_cells.SWEEPS, use_radar=False)
        radar = list(cache_cells.load_radar(s, args.cells, *tf.GATE, cache_cells.SWEEPS))
        vis = visibility.load(s, args.cells, base)
        moving = visibility.build_moving(s, args.cells, args.cand3d, *tf.GATE)
        rd = radar_candidates.build(s, args.cells, args.cand3d, args.cand2d)
        tf.DATA[s], tf.DATA["radar_" + s], tf.DATA["vis_" + s], tf.DATA["moving_" + s] = radar_candidates.augment(base, radar, vis, moving, rd)
        del rd
    ctx = mp.get_context("fork")
    for name in args.models.split(","):
        if name in results:
            continue
        if name == "SECOND":
            results[name] = dict(runs=[dict(seed=0, **{s: score(tf.DATA[s], [it["scores"] for it in tf.DATA[s]], VARIANTS[name])
                                                       for s in eval_splits})])
        else:
            cfg = tf.parse(name)
            for s in eval_splits:
                tf.DATA["ctx_" + s] = tf.gate_context(cfg, s, tf.DATA[s], tf.DATA["radar_" + s])
            with ctx.Pool(len(seeds)) as pool:
                results[name] = dict(runs=pool.map(run_seed, [(name, sd, eval_splits) for sd in seeds]))
        print(name, {s: {v: round(float(np.mean([r[s][v]["overall"]["50-200"]["APCD"] for r in results[name]["runs"]])), 4)
                         for v in VARIANTS[name]} for s in eval_splits}, flush=True)
        json.dump(results, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
