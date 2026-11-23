"""The ceiling of a radar detector (results/next_round/PLAN.md, Phase 5).

Scored cars beyond 50 m with no LiDAR point inside the box but at least one radar return in the gate are the cars a
radar detector could add. Their ground-truth boxes are injected as extra 3D candidates with a low score (S_LOW), with
every input a SECOND candidate has, and the saved fusion networks score the frames again:
- "none": the injected candidates are dropped (must reproduce the earlier numbers; scoring is per candidate);
- "low": they keep the score the model gives them;
- "perfect": their final score is set to 1 before NMS.

    python -m clocs_r.oracle_inject --splits select,val --out results/next_round/phase5/ceiling.json
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import pickle

import numpy as np
import torch

from clocs_r import aimotive, cache_cells, fusion, metrics, radar_features, range_switch, splits, train_fusion
from clocs_r import tune_fusion as tf
from clocs_r import visibility

S_LOW = 0.1  # s3D of an injected candidate
FAR_M = 50.0
FOUND_SCORE, FOUND_CD = 0.05, 2.0  # SECOND "already finds" a car: a candidate with this score within this distance (m)
SWITCH_M = 50.0
VARIANTS = ("none", "low", "perfect")
P1 = "runs/next_round/phase1"
MODELS = {  # name -> saved weights (None: SECOND alone)
    "SECOND": None,
    "R0+T2": f"{P1}/fusion_insample",
    "R2+T2": f"{P1}/fusion_insample",
    "R2+T2+K40": "runs/next_round/phase3/fusion",
    "R0+T2+K40+G": "runs/next_round/phase4/fusion",
    "R2+T2+K40+G": "runs/next_round/phase4/fusion",
}


def targets(gt, n_lidar, radar_returns, far_m=FAR_M):
    """Mask of the scored cars a radar detector could add: beyond far_m, no LiDAR point, at least one radar return."""
    if len(gt) == 0:
        return np.zeros(0, bool)
    return (np.hypot(gt[:, 0], gt[:, 1]) >= far_m) & (n_lidar == 0) & (radar_returns > 0)


def found_by(gt, boxes, scores, min_score=FOUND_SCORE, cd=FOUND_CD):
    """Mask of the cars with a candidate of at least min_score within cd metres of the center (BEV)."""
    strong = boxes[scores >= min_score]
    if len(gt) == 0 or len(strong) == 0:
        return np.zeros(len(gt), bool)
    return np.hypot(gt[:, None, 0] - strong[None, :, 0], gt[:, None, 1] - strong[None, :, 1]).min(1) <= cd


def inject_inputs(seq, frame, cand3d, cand2d, delta0, kappa, sweeps, s_low=S_LOW):
    """Per-car statistics of a frame and the inputs of its injected candidates, built as cache_cells and visibility do."""
    boxes, scores, det2d = cache_cells.load_candidates(seq, frame, cand3d, cand2d)
    gt, _, _ = cache_cells.eval_labels(seq, frame, boxes)
    lidar = seq.lidar(frame)
    pts, vel = seq.radar_accumulated(frame, sweeps)
    n_lidar = visibility.points_in_boxes(gt, lidar)
    rad = radar_features.candidate_features(gt, pts, vel, delta0, kappa)
    t = targets(gt, n_lidar, rad[:, 0])
    b = gt[t].astype(boxes.dtype)
    proj, valid = aimotive.project_boxes(seq, b)
    proj = proj.astype(np.float32)
    proj[~valid] = np.nan
    dist = (np.hypot(b[:, 0], b[:, 1]) / cache_cells.MAX_RANGE).astype(np.float32)
    cells, column = fusion.build_cells(det2d[:, :4], det2d[:, 4], proj, np.full(len(b), s_low, np.float32), dist,
                                       np.zeros((len(b), 0), np.float32))
    _, keep_eval, _ = cache_cells.eval_labels(seq, frame, b)
    stats = dict(range=np.hypot(gt[:, 0], gt[:, 1]).astype(np.float32), n_lidar=n_lidar, radar=rad[:, 0] > 0,
                 found=found_by(gt, boxes, scores), target=t)
    inject = dict(boxes=b, pairs=cells[:, :2].astype(np.float16), column=column.astype(np.int16), dist=dist,
                  radar=rad[t].astype(np.float16), vis=visibility.features(seq, b, lidar).astype(np.float16),
                  moving=visibility.moving_returns(b, pts, vel, delta0, kappa).astype(np.float16), keep_eval=keep_eval)
    return dict(stats=stats, inject=inject, key=(seq.name, frame), odd=seq.odd)


def _work(job):
    path, frame, a = job
    return inject_inputs(cache_cells._seq(path), frame, a["cand3d"], a["cand2d"], *a["gate"], a["sweeps"], a["s_low"])


def build(split, cells, cand3d, cand2d, gate=tf.GATE, sweeps=cache_cells.SWEEPS, s_low=S_LOW, workers=8):
    path = pathlib.Path(cells) / f"{split}_oracle_{cache_cells.radar_name(*gate, sweeps)}_s{s_low}.pkl"
    if not path.exists():
        a = dict(cand3d=cand3d, cand2d=cand2d, gate=gate, sweeps=sweeps, s_low=s_low)
        with mp.Pool(workers) as pool:
            out = pool.map(_work, [(s.path, f, a) for s, f in splits.frames(split)], chunksize=16)
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=4)
    with open(path, "rb") as f:
        return pickle.load(f)


def augment(base, radar, vis, moving, inj, s_low=S_LOW):
    """The inputs of a split with the injected candidates appended to each frame, after its own candidates.

    base: cache_cells items; radar: per-frame (n, 5) features or None; vis: visibility.load dict; moving: per-frame (n,)
    or None; inj: build() output, frame for frame. Each new item has an "injected" mask over its candidates.
    """
    assert [(it["seq"], it["frame"]) for it in base] == [x["key"] for x in inj], "injection and cells frames differ"
    out_base, out_radar, out_vis, out_moving = [], [], [], []
    for i, item in enumerate(base):
        j, n = inj[i]["inject"], len(item["scores"])
        k = len(j["boxes"])
        new = dict(item)
        new["boxes"] = np.concatenate([item["boxes"], j["boxes"].astype(item["boxes"].dtype)])
        new["scores"] = np.concatenate([item["scores"], np.full(k, s_low, item["scores"].dtype)])
        new["dist"] = np.concatenate([item["dist"], j["dist"].astype(item["dist"].dtype)])
        new["keep_eval"] = np.concatenate([item["keep_eval"], j["keep_eval"]])
        new["pairs"] = np.concatenate([item["pairs"], j["pairs"].astype(item["pairs"].dtype)])
        new["column"] = np.concatenate([item["column"].astype(np.int32), j["column"].astype(np.int32) + n])
        new["target"] = np.concatenate([item["target"], np.ones(k, item["target"].dtype)])
        new["weight"] = np.concatenate([item["weight"], np.ones(k, item["weight"].dtype)])
        new["injected"] = np.concatenate([np.zeros(n, bool), np.ones(k, bool)])
        out_base.append(new)
        if radar is not None:
            assert len(radar[i]) == n, (len(radar[i]), n)
            out_radar.append(np.concatenate([np.asarray(radar[i]), j["radar"].astype(np.asarray(radar[i]).dtype)]))
        out_vis.append(np.concatenate([np.asarray(vis["features"][i]), j["vis"].astype(np.asarray(vis["features"][i]).dtype)]))
        if moving is not None:
            out_moving.append(np.concatenate([np.asarray(moving[i]), j["moving"].astype(np.asarray(moving[i]).dtype)]))
    return (out_base, out_radar if radar is not None else None, dict(vis, features=out_vis),
            out_moving if moving is not None else None)


def variant_frames(base, scores, variant, switch_m=None):
    """(records for train_fusion.condition_groups, injected candidates kept by NMS) for one variant."""
    records, kept = [], 0
    for item, s in zip(base, scores):
        inj = item["injected"]
        s = np.array(s, np.float32)
        if switch_m is not None:
            s = range_switch.switch_scores(item["boxes"], s, item["scores"], switch_m)
        mask = item["keep_eval"].copy()
        if variant == "none":
            mask &= ~inj
        elif variant == "perfect":
            s[inj] = 1.0
        boxes, ss, flag = item["boxes"][mask], s[mask], inj[mask]
        keep = metrics.nms_bev(boxes, ss, 0.1)
        kept += int(flag[keep].sum())
        records.append((item["gt"], boxes[keep], ss[keep], item["odd"], item["seq"]))
    return records, kept


def score_variants(base, scores, switch_m=None):
    out = {}
    for v in VARIANTS:
        records, kept = variant_frames(base, scores, v, switch_m)
        out[v] = {name: metrics.evaluate([r[:3] for r in rows], bins=aimotive.DIST_BINS + [train_fusion.FAR])
                  for name, rows in train_fusion.condition_groups(records).items()}
        out[v]["injected_kept"] = kept
    return out


@torch.no_grad()
def net_scores(net, cfg, split, device):
    idx = tf.names_idx(cfg)
    extra, pairing = tf.p3_inputs(cfg, split)
    radar = tf.DATA["radar_" + split]
    out = []
    for i, item in enumerate(tf.DATA[split]):
        x, col = train_fusion.frame_cells(item, radar[i] if idx else None, idx, device, False,
                                          extra[i] if extra is not None else None, pairing[i] if pairing is not None else None)
        out.append(torch.sigmoid(net(x, col, len(item["scores"]))).cpu().numpy())
    return out


def run_seed(job):
    """In a forked child, as tune_fusion.run_seed."""
    name, seed, eval_splits = job
    torch.set_num_threads(1)
    device = torch.device("cuda")
    cfg = tf.parse(name)
    idx = tf.names_idx(cfg)
    path = pathlib.Path(MODELS[name]) / f"{name}_s{seed}.pth"
    state = torch.load(path, map_location="cpu")
    assert state["settings"] == cfg and state.get("far_weight") == 2.0 and tuple(state.get("gate", tf.GATE)) == tuple(tf.GATE), path
    net = tf.make_net(cfg, idx).to(device)
    net.load_state_dict(state["model"])
    net.eval()
    result = dict(seed=seed, weights=str(path))
    switch = {}
    for split in eval_splits:
        scores = net_scores(net, cfg, split, device)
        result[split] = score_variants(tf.DATA[split], scores)
        if name == "R2+T2":
            switch[split] = score_variants(tf.DATA[split], scores, SWITCH_M)
    return result, switch


def counts(inj):
    """Per condition and bin: scored cars beyond FAR_M, with no LiDAR point, targets, and targets SECOND already finds."""
    out = {}
    for odd in ("overall",) + tuple(splits.ODDS):
        rows = [x["stats"] for x in inj if odd == "overall" or x["odd"] == odd]
        if not rows:
            continue
        cat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        out[odd] = {}
        for lo, hi in ((50, 100), (100, 200), (50, 200)):
            b = (cat["range"] >= lo) & (cat["range"] < hi)
            no_lidar = b & (cat["n_lidar"] == 0)
            out[odd][f"{lo}-{hi}"] = dict(cars=int(b.sum()), no_lidar=int(no_lidar.sum()), radar=int((b & cat["radar"]).sum()),
                                          targets=int((b & cat["target"]).sum()), found=int((b & cat["target"] & cat["found"]).sum()),
                                          no_lidar_found=int((no_lidar & cat["found"]).sum()))
    return out


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
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.load(open(out)) if out.exists() else {}

    for split in eval_splits:
        inj = build(split, args.cells, args.cand3d, args.cand2d)
        results.setdefault("counts", {})[split] = counts(inj)
        base, _ = cache_cells.load(split, args.cells, *tf.GATE, cache_cells.SWEEPS, use_radar=False)
        radar = cache_cells.load_radar(split, args.cells, *tf.GATE, cache_cells.SWEEPS)
        vis = visibility.load(split, args.cells, base)
        moving = visibility.build_moving(split, args.cells, args.cand3d, *tf.GATE)
        (tf.DATA[split], tf.DATA["radar_" + split], tf.DATA["vis_" + split], tf.DATA["moving_" + split]) = \
            augment(base, radar, vis, moving, inj)
        print(split, "injected", sum(len(x["inject"]["boxes"]) for x in inj), "candidates", flush=True)
    json.dump(results, open(out, "w"), indent=1)

    ctx = mp.get_context("fork")
    for name in args.models.split(","):
        if name in results:
            continue
        if name == "SECOND":
            results[name] = dict(runs=[dict(seed=0, **{s: score_variants(tf.DATA[s], [it["scores"] for it in tf.DATA[s]])
                                                       for s in eval_splits})])
        else:
            cfg = tf.parse(name)
            if cfg.get("gated"):
                for s in eval_splits:
                    tf.DATA["ctx_" + s] = tf.gate_context(cfg, s, tf.DATA[s], tf.DATA["radar_" + s])
            with ctx.Pool(len(seeds)) as pool:
                runs = pool.map(run_seed, [(name, s, eval_splits) for s in seeds])
            results[name] = dict(runs=[r for r, _ in runs])
            if name == "R2+T2":
                results["switch(R2+T2)"] = dict(runs=[dict(seed=r["seed"], **sw) for r, sw in runs])
        far = {s: {v: round(float(np.mean([r[s][v]["overall"]["50-200"]["APCD"] for r in results[name]["runs"]])), 4)
                   for v in VARIANTS} for s in eval_splits}
        print(name, far, flush=True)
        json.dump(results, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
