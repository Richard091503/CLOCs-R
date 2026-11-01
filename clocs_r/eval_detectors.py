"""Score the trained SECOND and Cascade R-CNN on their own, from cached candidates, with no fusion or radar.

Usage: python -m clocs_r.eval_detectors --splits holdout,val --out results/detectors
Writes aimotive_second.json and aimotive_cascade.json, updating the entries for these splits.
"""
import argparse
import json
import multiprocessing as mp
import pathlib

import numpy as np

from clocs_r import aimotive, cache_cells, metrics, splits


def frame_3d(job):
    seq_path, frame, cand3d = job
    seq = cache_cells._seq(seq_path)
    with np.load(pathlib.Path(cand3d) / seq.name / f"{frame:07d}.npz") as z:
        boxes, scores = z["boxes"], z["scores"]
    gt, mask, _ = cache_cells.eval_labels(seq, frame, boxes)
    boxes, scores = boxes[mask], scores[mask]
    keep = metrics.nms_bev(boxes, scores, 0.1)
    return dict(gt=gt, det=boxes[keep], score=scores[keep], odd=seq.odd, seq=seq.name)


def frame_2d(job):
    seq_path, frame, cand2d = job
    seq = cache_cells._seq(seq_path)
    det = np.load(pathlib.Path(cand2d) / seq.name / f"{frame:07d}.npy")
    # Every car in view counts here, with no region limit: the camera also sees cars SECOND's grid leaves out
    cars, _ = seq.boxes(frame, splits.CAR_CLASSES)
    if len(cars):
        cars = cars[aimotive.in_camera_view(seq, cars)]
    rect, ok = aimotive.project_boxes(seq, cars)
    ignore, _ = seq.boxes(frame, splits.IGNORE_CLASSES)
    ignore_rect, ignore_ok = aimotive.project_boxes(seq, ignore)
    return dict(gt=rect[ok], gt_range=np.hypot(cars[ok, 0], cars[ok, 1]) if len(cars) else np.zeros(0), ignore=ignore_rect[ignore_ok],
                cand=det[:, :4], cand_scores=det[:, 4], odd=seq.odd, seq=seq.name)


def groups(records):
    """Named subsets of a split's frames: all of them, each driving condition, and no tunnels."""
    out = {"overall": records}
    for odd in splits.ODDS:
        out[odd] = [r for r in records if r["odd"] == odd]
    out["no_tunnels"] = [r for r in records if r["seq"] not in splits.TUNNEL_SCENES]
    return {name: rows for name, rows in out.items() if rows}


def update(path, split, entry):
    results = json.load(open(path)) if path.exists() else {}
    results[split] = entry
    with open(path, "w") as f:
        json.dump(results, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="holdout,val")
    ap.add_argument("--out", default="results/detectors")
    ap.add_argument("--cand3d", default="runs/candidates3d")
    ap.add_argument("--cand2d", default="runs/candidates2d")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits.split(","):
        items = [(seq.path, f) for seq, f in splits.frames(split)]
        with mp.Pool(args.workers) as pool:
            rec3d = pool.map(frame_3d, [(s, f, args.cand3d) for s, f in items], chunksize=16)
            rec2d = pool.map(frame_2d, [(s, f, args.cand2d) for s, f in items], chunksize=16)
        second = {name: metrics.evaluate([(r["gt"], r["det"], r["score"]) for r in rows]) for name, rows in groups(rec3d).items()}
        cascade = {name: metrics.evaluate_2d(rows) for name, rows in groups(rec2d).items()}
        second["frames"], cascade["frames"] = len(items), len(items)
        update(out / "aimotive_second.json", split, second)
        update(out / "aimotive_cascade.json", split, cascade)
        print(split, "SECOND", {k: {m: round(v[m], 3) for m in ("n_gt", "AP3D@0.5", "AP3D@0.7", "APBEV@0.3", "APCD", "recall@BEV0.3")}
                                for k, v in second["overall"].items()}, flush=True)
        print(split, "Cascade AP", cascade["overall"]["ap"], flush=True)
        print(split, "Cascade candidate recall", cascade["overall"]["recall"], flush=True)


if __name__ == "__main__":
    main()
