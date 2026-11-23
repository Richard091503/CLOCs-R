"""Phase 1: frames and scored cars by condition and distance in each split, from the labels only.

    python results/next_round/phase1_counts.py     # prints markdown, writes results/next_round/phase1/counts.json
"""
import collections
import json
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from clocs_r import aimotive, splits

OUT = pathlib.Path(__file__).parent / "phase1"
SPLITS = ("train", "holdout", "select", "fit", "val")
BINS = ["0-30", "30-50", "50-100", "100-200"]


def scene(path):
    seq = aimotive.Sequence(path)
    c = np.zeros(len(BINS), int)
    for f in seq.kept:
        gt, _ = seq.boxes(f, splits.CAR_CLASSES)
        if len(gt):
            d = np.hypot(*gt[aimotive.in_region(gt) & aimotive.in_camera_view(seq, gt)][:, :2].T)
            c += [int(((d >= a) & (d < b)).sum()) for a, b in aimotive.DIST_BINS]
    return path, seq.odd, len(seq.kept), c.tolist()


def main():
    paths = sorted({p for s in SPLITS for p in splits.scenes(s)})
    with ProcessPoolExecutor(8) as pool:
        per = {p: r for p, *r in pool.map(scene, paths, chunksize=4)}
    table = {}
    for s in SPLITS:
        t = collections.defaultdict(lambda: dict(scenes=0, recordings=set(), frames=0, cars=np.zeros(len(BINS), int)))
        for p in splits.scenes(s):
            odd, n, c = per[p]
            for g in (odd, "all"):
                t[g]["scenes"] += 1
                t[g]["recordings"].add(splits.recording(p))
                t[g]["frames"] += n
                t[g]["cars"] += c
        table[s] = {g: dict(scenes=v["scenes"], recordings=len(v["recordings"]), frames=v["frames"],
                            cars=dict(zip(BINS, v["cars"].tolist()))) for g, v in t.items()}
    json.dump(table, open(OUT / "counts.json", "w"), indent=1)

    print("| Condition | Split | Scenes | Recordings | Frames | 0-30 m | 30-50 m | 50-100 m | 100-200 m | 100-200 m share of cars |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for g in ("all",) + splits.ODDS:
        for s in SPLITS:
            v = table[s].get(g)
            if v is None:
                continue
            c = v["cars"]
            tot = sum(c.values())
            print(f"| {g} | {s} | {v['scenes']} | {v['recordings']} | {v['frames']:,} | " +
                  " | ".join(f"{c[b]:,}" for b in BINS) + f" | {c['100-200'] / max(tot, 1):.0%} |")


if __name__ == "__main__":
    main()
