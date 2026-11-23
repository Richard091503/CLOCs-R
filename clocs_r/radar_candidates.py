"""The radar detector scored on its own (--detector), and its matching constants (RD_MIN, MATCH_M) used by radar_pairs.

The rest of this module builds radar-sourced candidates (model RD, docs/MODELS.md; its fusion code is on the thesis
branch); detector_report reuses that cache to score the detector's own boxes.

Radar detections are the exported radar PointPillars boxes (export_3d --nms) with score >= RD_MIN.
- Every SECOND candidate gets rd: the highest radar-detection score within MATCH_M of its BEV center, or 0.
- A radar detection with no SECOND candidate of score >= RD_MIN within MATCH_M becomes a new candidate: s3D is its
  radar score, rd the same score, src 1, and its radar features, visibility inputs, moving returns and 2D pairs are
  built as for SECOND candidates. Its training target is 1 within MATCH_M of a scored car, ignored up to IGNORE_M.
The radar channels gain two columns, rd and src (RD_COLUMNS), after the five features.

    python -m clocs_r.radar_candidates --split fit,select,val            # caches beside the cells
    python -m clocs_r.radar_candidates --detector select,val --dets runs/lcr/d2_dets --tag d2 --out results/lcr/detector_d2.json
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import pickle

import numpy as np

from clocs_r import aimotive, cache_cells, fusion, metrics, radar_features, splits, train_fusion, visibility

RD_MIN = 0.05
MATCH_M, IGNORE_M = 2.0, 4.0
RD_COLUMNS = (5, 6)  # rd, src in the radar array of an RD configuration
GATE = (0.5, 0.02)
P1 = "runs/lcr"
DETS = "runs/lcr/d2_dets"


def nearest(a, b):
    """For each BEV center in a (N, >=2), the distance to the nearest center in b (M, >=2); inf if b is empty."""
    if len(b) == 0:
        return np.full(len(a), np.inf)
    if len(a) == 0:
        return np.zeros(0)
    return np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]).min(1)


def match(boxes, scores, rboxes, rscores, rd_min=RD_MIN, match_m=MATCH_M):
    """rd for every SECOND candidate, and the mask of radar detections that become new candidates."""
    det = rscores >= rd_min
    rb, rs = rboxes[det], rscores[det]
    rd = np.zeros(len(boxes), np.float32)
    if len(rb) and len(boxes):
        d = np.hypot(boxes[:, None, 0] - rb[None, :, 0], boxes[:, None, 1] - rb[None, :, 1])
        rd = np.where(d <= match_m, rs[None, :], 0.0).max(1).astype(np.float32)
    new = np.zeros(len(rboxes), bool)
    new[np.nonzero(det)[0]] = nearest(rb, boxes[scores >= rd_min]) > match_m
    return rd, new


def targets(new_boxes, gt, on_ignored, match_m=MATCH_M, ignore_m=IGNORE_M):
    """Training target and weight of radar-sourced candidates, by BEV center distance to the scored cars."""
    d = nearest(new_boxes, gt)
    target = (d <= match_m).astype(np.float32)
    weight = np.where((d > match_m) & (d <= ignore_m), 0.0, 1.0).astype(np.float32)
    weight[on_ignored] = 0.0
    return target, weight


def rd_inputs(seq, frame, cand3d, cand2d, dets, gate=GATE, sweeps=cache_cells.SWEEPS):
    boxes, scores, det2d = cache_cells.load_candidates(seq, frame, cand3d, cand2d)
    with np.load(pathlib.Path(dets) / seq.name / f"{frame:07d}.npz") as z:
        rboxes, rscores = z["boxes"].astype(np.float32), z["scores"].astype(np.float32)
    rd, new = match(boxes, scores, rboxes, rscores)
    b, s = rboxes[new], rscores[new]
    proj, valid = aimotive.project_boxes(seq, b)
    proj = proj.astype(np.float32)
    proj[~valid] = np.nan
    dist = (np.hypot(b[:, 0], b[:, 1]) / cache_cells.MAX_RANGE).astype(np.float32)
    cells, column = fusion.build_cells(det2d[:, :4], det2d[:, 4], proj, s, dist, np.zeros((len(b), 0), np.float32))
    gt, _, _ = cache_cells.eval_labels(seq, frame, boxes)
    _, keep_eval, on_ignored = cache_cells.eval_labels(seq, frame, b)
    target, weight = targets(b, gt, on_ignored)
    pts, vel = seq.radar_accumulated(frame, sweeps)
    lidar = seq.lidar(frame)
    _, det_keep, _ = cache_cells.eval_labels(seq, frame, rboxes)  # for the radar detector scored on its own
    inject = dict(boxes=b, scores=s, pairs=cells[:, :2].astype(np.float16), column=column.astype(np.int16), dist=dist,
                  radar=radar_features.candidate_features(b, pts, vel, *gate).astype(np.float16),
                  vis=visibility.features(seq, b, lidar).astype(np.float16),
                  moving=visibility.moving_returns(b, pts, vel, *gate).astype(np.float16),
                  keep_eval=keep_eval, target=target, weight=weight)
    return dict(key=(seq.name, frame), rd=rd.astype(np.float16), inject=inject,
                dets=dict(boxes=rboxes, scores=rscores, keep_eval=det_keep))


def _work(job):
    path, frame, a = job
    return rd_inputs(cache_cells._seq(path), frame, a["cand3d"], a["cand2d"], a["dets"])


def cache_path(split, cells, tag=""):
    """tag: the detector, for detectors after Phase 6's (Phase 6b); empty for Phase 6's, which keeps its file name."""
    return pathlib.Path(cells) / f"{split}_rd_{cache_cells.radar_name(*GATE, cache_cells.SWEEPS)}_min{RD_MIN}{'_' + tag if tag else ''}.pkl"


def build(split, cells, cand3d, cand2d, dets=DETS, workers=8, tag=""):
    path = cache_path(split, cells, tag)
    if not path.exists():
        a = dict(cand3d=cand3d, cand2d=cand2d, dets=dets)
        with mp.Pool(workers) as pool:
            out = pool.map(_work, [(s.path, f, a) for s, f in splits.frames(split)], chunksize=16)
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=4)
    with open(path, "rb") as f:
        return pickle.load(f)


def augment(base, radar, vis, moving, rd):
    """Appends each frame's radar-sourced candidates, in place in the lists given (frees the old items as it goes).

    radar: per-frame (n, 5) features, returned as (n + k, 7) with rd and src; vis: visibility.load dict (features
    replaced); moving: per-frame (n,). Each item gains "src" (and "injected", the same mask, for oracle_inject's scoring).
    """
    assert [(it["seq"], it["frame"]) for it in base] == [x["key"] for x in rd], "radar candidates and cells frames differ"
    radar_out = [None] * len(base)
    for i, item in enumerate(base):
        j, n = rd[i]["inject"], len(item["scores"])
        k = len(j["boxes"])
        new = dict(item)
        new["boxes"] = np.concatenate([item["boxes"], j["boxes"].astype(item["boxes"].dtype)])
        new["scores"] = np.concatenate([item["scores"], j["scores"].astype(item["scores"].dtype)])
        new["dist"] = np.concatenate([item["dist"], j["dist"].astype(item["dist"].dtype)])
        new["keep_eval"] = np.concatenate([item["keep_eval"], j["keep_eval"]])
        new["pairs"] = np.concatenate([item["pairs"], j["pairs"].astype(item["pairs"].dtype)])
        new["column"] = np.concatenate([item["column"].astype(np.int32), j["column"].astype(np.int32) + n])
        new["target"] = np.concatenate([item["target"], j["target"].astype(item["target"].dtype)])
        new["weight"] = np.concatenate([item["weight"], j["weight"].astype(item["weight"].dtype)])
        new["src"] = new["injected"] = np.concatenate([np.zeros(n, bool), np.ones(k, bool)])
        base[i] = new
        r = np.asarray(radar[i])
        assert len(r) == n, (len(r), n)
        top = np.concatenate([r, rd[i]["rd"].astype(r.dtype)[:, None], np.zeros((n, 1), r.dtype)], 1)
        bottom = np.concatenate([j["radar"].astype(r.dtype), j["scores"].astype(r.dtype)[:, None], np.ones((k, 1), r.dtype)], 1)
        radar_out[i] = np.concatenate([top, bottom])
        vis["features"][i] = np.concatenate([np.asarray(vis["features"][i]), j["vis"].astype(np.asarray(vis["features"][i]).dtype)])
        if moving is not None:
            moving[i] = np.concatenate([np.asarray(moving[i]), j["moving"].astype(np.asarray(moving[i]).dtype)])
    return base, radar_out, vis, moving


def detector_report(split, cells, base, dets=DETS, tag="", cand3d=None, cand2d=None):
    """The radar detector on its own: metrics by distance bin and condition, as train_fusion.evaluate reports them."""
    rd = build(split, cells, cand3d, cand2d, dets, tag=tag)
    records = []
    for item, x in zip(base, rd):
        d = x["dets"]
        m = d["keep_eval"]
        keep = metrics.nms_bev(d["boxes"][m], d["scores"][m], 0.1)
        records.append((item["gt"], d["boxes"][m][keep], d["scores"][m][keep], item["odd"], item["seq"]))
    return {name: metrics.evaluate([r[:3] for r in rows], bins=aimotive.DIST_BINS + [train_fusion.FAR])
            for name, rows in train_fusion.condition_groups(records).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="")
    ap.add_argument("--detector", default="", help="splits on which to score the radar detector alone")
    ap.add_argument("--cells", default=f"{P1}/cells")
    ap.add_argument("--cand3d", default=f"{P1}/cand/3d")
    ap.add_argument("--cand2d", default=f"{P1}/cand/2d")
    ap.add_argument("--dets", default=DETS)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="", help="detector tag for the caches (Phase 6b/7: d2, d3); empty for Phase 6's detector")
    args = ap.parse_args()
    for split in filter(None, args.split.split(",")):
        rd = build(split, args.cells, args.cand3d, args.cand2d, args.dets, tag=args.tag)
        k = [len(x["inject"]["boxes"]) for x in rd]
        pos = sum(int(x["inject"]["target"].sum()) for x in rd)
        print(f"{split}: {sum(k)} radar-sourced candidates ({np.mean(k):.2f} per frame), {pos} within {MATCH_M} m of a scored car; "
              f"SECOND candidates with a radar match: {sum(int((x['rd'] > 0).sum()) for x in rd)}", flush=True)
    if args.detector:
        out = {}
        for split in args.detector.split(","):
            base, _ = cache_cells.load(split, args.cells, *GATE, cache_cells.SWEEPS, use_radar=False)
            out[split] = detector_report(split, args.cells, base, args.dets, args.tag, args.cand3d, args.cand2d)
            print(split, {b: round(out[split]["overall"][b]["APCD"], 4) for b in ("all", "50-100", "100-200", "50-200")}, flush=True)
            del base
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
