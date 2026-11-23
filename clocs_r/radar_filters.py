"""Precision filters for radar-only candidates, appended to a base model's output (results/next_round/PLAN.md, Phase 6b).

Radar-only candidates are those of radar_candidates (a radar detection with score >= RD_MIN and no SECOND candidate of
score >= RD_MIN within 2 m). A filter is a score threshold and a cue: none, moving (a moving return in the gate), lidar
(a LiDAR point in the box), persist (a detection of the same detector in the previous frame within 2 m, moved by the
ego poses) or moving|persist. Kept candidates are merged with the base model's boxes after its NMS, scored either
"last" (below every base box of the split, so AP cannot fall) or "calibrated" (the filter's precision on fit in the
candidate's distance bin), and BEV NMS runs again.

    python -m clocs_r.radar_filters --dets runs/next_round/phase6/radar_dets --tag d1 --out results/next_round/phase6b/d1_select.json
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import pickle

import numpy as np
import torch

from clocs_r import cache_cells, metrics, radar_candidates, splits, train_fusion
from clocs_r import oracle_inject as oi
from clocs_r import tune_fusion as tf
from clocs_r import visibility

THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)
CUES = ("none", "moving", "lidar", "persist", "moving|persist")
MODES = ("last", "calibrated")
CAL_BINS = ((0, 50), (50, 100), (100, 1e9))
FAR_BINS = [(50, 100), (100, 200), (50, 200)]
P1 = "runs/next_round/phase1"
BASES = {"SECOND": None, "R2+T2+K40+G": "runs/next_round/phase4/fusion"}


def filters():
    return [(t, c) for t in THRESHOLDS for c in CUES]


def fname(t, c):
    return f"t{t}|{c}"


def persist(seq, frame, dets, boxes, rd_min=radar_candidates.RD_MIN, match_m=radar_candidates.MATCH_M):
    """Mask of boxes with a detection of score >= rd_min in the previous frame within match_m, after the ego motion."""
    p = pathlib.Path(dets) / seq.name / f"{frame - 1:07d}.npz"
    poses = seq.meta["poses"]
    if len(boxes) == 0 or not p.exists() or frame not in poses or frame - 1 not in poses:
        return np.zeros(len(boxes), bool)
    with np.load(p) as z:
        prev = z["boxes"][z["scores"] >= rd_min]
    if len(prev) == 0:
        return np.zeros(len(boxes), bool)
    T = np.linalg.inv(poses[frame]) @ poses[frame - 1]
    xy = prev[:, :2] @ T[:2, :2].T + T[:2, 3]
    return radar_candidates.nearest(boxes, xy) <= match_m


def _cue_work(job):
    path, frame, dets, inj = job
    seq = cache_cells._seq(path)
    return persist(seq, frame, dets, inj["boxes"])


def cues(split, cells, dets, tag, rd):
    """Per frame: the radar-only candidates with their cues and truth, cached as {split}_cues_{tag}.pkl."""
    path = pathlib.Path(cells) / f"{split}_cues_{tag}.pkl"
    if not path.exists():
        frames = splits.frames(split)
        assert [(s.name, f) for s, f in frames] == [x["key"] for x in rd]
        with mp.Pool(8) as pool:
            pers = pool.map(_cue_work, [(s.path, f, dets, x["inject"]) for (s, f), x in zip(frames, rd)], chunksize=16)
        out = []
        for x, p in zip(rd, pers):
            j = x["inject"]
            out.append(dict(boxes=j["boxes"], scores=j["scores"], keep_eval=j["keep_eval"], true=j["target"] > 0,
                            moving=np.asarray(j["moving"], np.float32) > 0, lidar=np.asarray(j["vis"], np.float32)[:, 1] > 0,
                            persist=p))
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=4)
    with open(path, "rb") as f:
        return pickle.load(f)


def mask(c, t, cue):
    m = c["keep_eval"] & (c["scores"] >= t)
    if cue == "moving":
        m &= c["moving"]
    elif cue == "lidar":
        m &= c["lidar"]
    elif cue == "persist":
        m &= c["persist"]
    elif cue == "moving|persist":
        m &= c["moving"] | c["persist"]
    return m


def _bin(boxes, bins=CAL_BINS):
    r = np.hypot(boxes[:, 0], boxes[:, 1])
    return np.searchsorted([b[1] for b in bins[:-1]], r, side="right")


def precision_table(cs):
    """Kept and true radar-only candidates per filter and distance bin, and the precision."""
    out = {}
    for t, cue in filters():
        kept, true = np.zeros(len(CAL_BINS), int), np.zeros(len(CAL_BINS), int)
        for c in cs:
            m = mask(c, t, cue)
            if m.any():
                b = _bin(c["boxes"][m])
                np.add.at(kept, b, 1)
                np.add.at(true, b, c["true"][m].astype(int))
        out[fname(t, cue)] = dict(kept=kept.tolist(), true=true.tolist(),
                                  precision=[float(tr / k) if k else 0.0 for tr, k in zip(true, kept)])
    return out


def apcd(frames, bins=FAR_BINS):
    """APCD exactly as metrics.evaluate computes it (same matching and AP), for "all" and the given bins only."""
    names = ["all"] + [f"{lo}-{hi}" for lo, hi in bins]
    ranges = [(0.0, float("inf"))] + list(bins)
    st = {n: dict(n_gt=0, scores=[], cd={t: [] for t in metrics.CD_THRESHOLDS}) for n in names}
    for gt, det, score in frames:
        order = np.argsort(-score)
        det, score = det[order], score[order]
        rg, rdd = np.hypot(gt[:, 0], gt[:, 1]), np.hypot(det[:, 0], det[:, 1])
        for name, (lo, hi) in zip(names, ranges):
            mg, md = (rg >= lo) & (rg < hi), (rdd >= lo) & (rdd < hi)
            g, d = gt[mg], det[md]
            s = st[name]
            s["n_gt"] += len(g)
            s["scores"].append(score[md])
            dc = np.hypot(g[:, None, 0] - d[None, :, 0], g[:, None, 1] - d[None, :, 1]) if len(g) and len(d) else np.zeros((len(g), len(d)))
            for t in metrics.CD_THRESHOLDS:
                s["cd"][t].append(metrics._match(g, d, dc, False, t)[0])
    out = {}
    for name in names:
        s = st[name]
        scores = np.concatenate(s["scores"])
        order = np.argsort(-scores)
        out[name] = float(np.mean([metrics._ap(np.concatenate(s["cd"][t])[order], s["n_gt"]) for t in metrics.CD_THRESHOLDS]))
    return out


def base_frames(base, scores):
    """The base model's kept boxes after its NMS: (gt, boxes, scores, odd, seq) per frame."""
    out = []
    for item, s in zip(base, scores):
        m = item["keep_eval"]
        b, ss = item["boxes"][m], np.asarray(s, np.float32)[m]
        keep = metrics.nms_bev(b, ss, 0.1)
        out.append((item["gt"], b[keep], ss[keep], item["odd"], item["seq"]))
    return out


def merged(bf, cs, t, cue, mode, calib=None):
    """The base frames with the filtered radar-only candidates added, after BEV NMS again."""
    floor = min(float(f[2].min()) for f in bf if len(f[2])) * 0.5  # "last": below every base box of the split
    out = []
    for (gt, b, s, odd, seq), c in zip(bf, cs):
        m = mask(c, t, cue)
        if not m.any():
            out.append((gt, b, s, odd, seq))
            continue
        rb = c["boxes"][m]
        rs = c["scores"][m] * floor if mode == "last" else np.asarray(calib, np.float32)[_bin(rb)]
        boxes, scores = np.concatenate([b, rb.astype(b.dtype)]), np.concatenate([s, rs.astype(s.dtype)])
        keep = metrics.nms_bev(boxes, scores, 0.1)
        out.append((gt, boxes[keep], scores[keep], odd, seq))
    return out


def select_table(bf, cs, calib):
    """APCD (all, 50-100, 100-200, 50-200) of the base alone and with every filter and mode."""
    res = {"base": apcd([f[:3] for f in bf])}
    for t, cue in filters():
        for mode in MODES:
            mf = merged(bf, cs, t, cue, mode, calib[fname(t, cue)]["precision"] if mode == "calibrated" else None)
            res[f"{fname(t, cue)}|{mode}"] = apcd([f[:3] for f in mf])
    return res


def _seed_job(job):
    base_name, seed, split = job
    torch.set_num_threads(1)
    if base_name == "SECOND":
        scores = [it["scores"] for it in tf.DATA[split]]
    else:
        cfg = tf.parse(base_name)
        idx = tf.names_idx(cfg)
        state = torch.load(pathlib.Path(BASES[base_name]) / f"{base_name}_s{seed}.pth", map_location="cpu")
        assert state["settings"] == cfg
        net = tf.make_net(cfg, idx).cuda()
        net.load_state_dict(state["model"])
        net.eval()
        scores = oi.net_scores(net, cfg, split, torch.device("cuda"))
    bf = base_frames(tf.DATA[split], scores)
    return dict(seed=seed, **select_table(bf, tf.DATA["cues_" + split], tf.DATA["calib"]))


def load_split(split, cells, cand3d, base_name="R2+T2+K40+G"):
    tf.DATA[split], _ = cache_cells.load(split, cells, *tf.GATE, cache_cells.SWEEPS, use_radar=False)
    tf.DATA["radar_" + split] = cache_cells.load_radar(split, cells, *tf.GATE, cache_cells.SWEEPS)
    tf.DATA["vis_" + split] = visibility.load(split, cells, tf.DATA[split])
    tf.DATA["moving_" + split] = visibility.build_moving(split, cells, cand3d, *tf.GATE)
    cfg = tf.parse(base_name)
    tf.DATA["ctx_" + split] = tf.gate_context(cfg, split, tf.DATA[split], tf.DATA["radar_" + split])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True, help="exported radar-detector boxes")
    ap.add_argument("--tag", required=True, help="detector name for the caches (d1, d2, d3)")
    ap.add_argument("--split", default="select")
    ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--cells", default=f"{P1}/cells_in")
    ap.add_argument("--cand3d", default=f"{P1}/cand_in/3d")
    ap.add_argument("--cand2d", default=f"{P1}/cand_in/2d")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    split = args.split
    rd_tag = "" if args.tag == "d1" else args.tag  # D1's radar-candidate caches are Phase 6's
    out = dict(detector=args.tag, dets=args.dets)
    for s in ("fit", split):
        rd = radar_candidates.build(s, args.cells, args.cand3d, args.cand2d, args.dets, tag=rd_tag)
        tf.DATA["cues_" + s] = cues(s, args.cells, args.dets, args.tag, rd)
        del rd
        out["precision_" + s] = precision_table(tf.DATA["cues_" + s])
        print(s, "cues ready", flush=True)
    tf.DATA["calib"] = out["precision_fit"]
    del tf.DATA["cues_fit"]
    load_split(split, args.cells, args.cand3d)
    jobs = [("SECOND", 0, split)] + [("R2+T2+K40+G", int(s), split) for s in args.seeds.split(",")]
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        runs = pool.map(_seed_job, jobs)
    out["SECOND"] = runs[0]
    out["R2+T2+K40+G"] = runs[1:]
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    best = max((k for k in runs[1] if k not in ("seed", "base")),
               key=lambda k: np.mean([r[k]["50-200"] - r["base"]["50-200"] for r in runs[1:]]))
    print("best on", split, best, np.mean([r[best]["50-200"] - r["base"]["50-200"] for r in runs[1:]]), flush=True)


if __name__ == "__main__":
    main()
