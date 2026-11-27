"""fit_sub: whole fit scenes, drawn per driving condition, as many frames as select has in that condition.

Rule (results/fit_sub_control/PREDICTIONS.md), fixed before any training:
- per condition, fit's scenes in clocs_r.splits order are shuffled by a fresh np.random.RandomState(0);
- scenes are taken in that order, and the draw stops at the prefix whose kept-frame count is closest to select's
  count for the condition (the first such prefix; frame counts only grow, so this is where adding the next scene
  would first move the count further from the target).
Far cars: scored ground-truth cars (cache_cells.eval_labels: region and front camera view) by BEV range, in the
metric's bins 50-100 and 100-200 m, over the kept frames. Writes results/fit_sub_control/subset.json.

    python results/fit_sub_control/draw_subset.py
"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from clocs_r import aimotive, cache_cells, splits  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "subset.json"
SEED = 0


def condition(scene):
    return scene.rstrip(os.sep).split(os.sep)[-2]


def name(scene):
    return os.path.basename(scene.rstrip(os.sep))


def describe(scenes):
    """Frames, recordings and far cars of a list of scenes, per condition and in all."""
    out = {}
    for s in scenes:
        seq = aimotive.Sequence(s)
        c = out.setdefault(condition(s), dict(scenes=0, frames=0, recordings=set(), cars_50_100=0, cars_100_200=0))
        c["scenes"] += 1
        c["frames"] += len(seq.kept)
        c["recordings"].add(splits.recording(s))
        for f in seq.kept:
            gt, _, _ = cache_cells.eval_labels(seq, f, np.zeros((0, 7), np.float32))
            r = np.hypot(gt[:, 0], gt[:, 1]) if len(gt) else np.zeros(0)
            c["cars_50_100"] += int(((r >= 50) & (r < 100)).sum())
            c["cars_100_200"] += int(((r >= 100) & (r < 200)).sum())
    total = dict(scenes=0, frames=0, recordings=set(), cars_50_100=0, cars_100_200=0)
    for c in out.values():
        for k in ("scenes", "frames", "cars_50_100", "cars_100_200"):
            total[k] += c[k]
        total["recordings"] |= c["recordings"]
    out = {k: out[k] for k in splits.ODDS if k in out}
    out["all"] = total
    for c in out.values():
        c["recordings"] = len(c["recordings"])
        c["cars_50_200"] = c["cars_50_100"] + c["cars_100_200"]
    return out


def main():
    fit = splits.scenes("fit")
    select = splits.scenes("select")
    target = {c: sum(len(aimotive.Sequence(s).kept) for s in select if condition(s) == c) for c in splits.ODDS}
    chosen, draw = [], {}
    for c in splits.ODDS:
        pool = [s for s in fit if condition(s) == c]
        order = np.random.RandomState(SEED).permutation(len(pool))
        cum = np.cumsum([len(aimotive.Sequence(pool[i]).kept) for i in order])
        k = int(np.argmin(np.abs(cum - target[c]))) + 1
        picked = [pool[i] for i in order[:k]]
        chosen += picked
        draw[c] = dict(fit_scenes=len(pool), target_frames=target[c], scenes_drawn=k, frames=int(cum[k - 1]),
                       frames_minus_target=int(cum[k - 1] - target[c]),
                       next_scene_frames_minus_target=int(cum[k] - target[c]) if k < len(cum) else None)
    chosen_set = set(chosen)
    scenes = [name(s) for s in fit if s in chosen_set]  # fit's order, as the cached cells
    out = dict(rule=__doc__.split("\n\n")[0] + " Seed 0, a fresh RandomState per condition; first prefix closest to "
               "select's kept frames.", seed=SEED, draw=draw, scenes=scenes,
               recordings=sorted({splits.recording(s) for s in scenes}),
               fit_sub=describe([s for s in fit if s in chosen_set]), select=describe(select), fit=describe(fit))
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("draw", "fit_sub", "select", "fit")}, indent=1))
    print(len(scenes), "scenes from", len(out["recordings"]), "recordings; wrote", OUT)


if __name__ == "__main__":
    main()
