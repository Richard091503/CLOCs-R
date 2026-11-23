"""Review round, task 2: is SECOND more reliable on its own training scenes (fit) than on new ones (select, val)?

1. SECOND alone, APCD on fit, select and val (cached candidates, eval_detectors.frame_3d: region and camera view, trucks
   and buses removed, BEV NMS 0.1).
2. Reliability of SECOND's candidates beyond 50 m after that NMS: per score bin, the share that is true (BEV centre
   within 2 m of a scored car).
3. The fit-trained Cascade R-CNN's candidate recall@0.5 for cars at 100-200 m on fit (in-sample), select and val
   (eval_detectors.frame_2d, metrics.evaluate_2d).
The select and val numbers must equal the stored results/lcr/data/aimotive_second.json and aimotive_cascade.json.

    cd ~/Documents/CLOCs-R && source env.sh
    python ../CLOCs-R-review/results/review_round/second_reliability.py --out ../CLOCs-R-review/results/review_round/second_reliability.json
"""
import argparse
import json
import multiprocessing as mp

import numpy as np

from clocs_r import aimotive, eval_detectors, metrics, splits
from clocs_r.train_fusion import FAR

CAND = "runs/next_round/phase1/cand_in"
SCORE_BINS = [(0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
TRUE_M = 2.0


def reliability(rec3d):
    """{score bin: (n, share true)} for kept candidates beyond 50 m."""
    sc, ok = [], []
    for r in rec3d:
        d, s, g = r["det"], r["score"], r["gt"]
        far = np.hypot(d[:, 0], d[:, 1]) >= 50
        d, s = d[far], s[far]
        if not len(d):
            continue
        dist = (np.hypot(d[:, None, 0] - g[None, :, 0], d[:, None, 1] - g[None, :, 1]).min(1) if len(g)
                else np.full(len(d), np.inf))
        sc.append(s)
        ok.append(dist <= TRUE_M)
    sc, ok = np.concatenate(sc), np.concatenate(ok)
    out = {}
    for lo, hi in SCORE_BINS:
        m = (sc >= lo) & ((sc < hi) if hi < 1.0 else (sc <= hi))
        out[f"{lo}-{hi}"] = dict(n=int(m.sum()), true=float(ok[m].mean()) if m.any() else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="select,val,fit")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {}
    for split in args.splits.split(","):
        items = [(seq.path, f) for seq, f in splits.frames(split)]
        with mp.Pool(args.workers) as pool:
            rec3d = pool.map(eval_detectors.frame_3d, [(s, f, f"{CAND}/3d") for s, f in items], chunksize=16)
            rec2d = pool.map(eval_detectors.frame_2d, [(s, f, f"{CAND}/2d") for s, f in items], chunksize=16)
        bins = aimotive.DIST_BINS + [FAR]
        second = metrics.evaluate([(r["gt"], r["det"], r["score"]) for r in rec3d], bins=bins)
        cascade = metrics.evaluate_2d(rec2d, bins=bins)
        out[split] = dict(frames=len(items), second_apcd={b: v["APCD"] for b, v in second.items()},
                          second_n_gt={b: v["n_gt"] for b, v in second.items()},
                          reliability_far=reliability(rec3d),
                          cascade_recall05={b: v["recall@0.5"] for b, v in cascade["recall"].items()})
        print(split, json.dumps(out[split]), flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
