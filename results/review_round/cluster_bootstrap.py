"""Review round, task 1: scene-level and recording-level (cluster) bootstrap of APCD differences.

The scene level is exactly the thesis branch's clocs_r/bootstrap.py: 1,000 resamples of scenes with replacement,
RandomState(0), seeds averaged inside each resample, percentile 95% intervals. The recording level draws recordings
(clocs_r.splits.recording) with replacement and keeps all their scenes; everything else is the same. Both use the same
per-frame matches, computed once per model and seed with the thesis code (bootstrap.kept_frames, frame_matches,
weighted_apcd), so the scene level reproduces the stored results/next_round/phase7/bootstrap_*.json.

L0, LRf and B can only be scored with the thesis code, so this runs with the thesis worktree on PYTHONPATH and the
master checkout (which holds runs/) as the working directory:

    cd ~/Documents/CLOCs-R && source ../CLOCs-R-thesis/env.sh
    python ../CLOCs-R-review/results/review_round/cluster_bootstrap.py --split select --out ../CLOCs-R-review/results/review_round/bootstrap_select.json
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from clocs_r import bootstrap as bs
from clocs_r import selector, splits

THESIS = pathlib.Path.home() / "Documents" / "CLOCs-R-thesis"
DETS = "runs/next_round/phase7/d2_dets"  # results/next_round/phase7/rp_detector.txt
# [label, kind, config, weights]: exactly the model definitions of thesis phase7_val.sh, phase7_lcr.sh and phase7_lcrd.sh
MODELS = [
    ["L", "second", "", ""],
    ["L0", "fusion", "R0+T2+NC", "runs/next_round/phase7/fusion"],
    ["LRf", "fusion", "R2+T2+NC", "runs/next_round/phase7/fusion"],
    ["LRd", "fusion", "R0+T2+RP", "runs/next_round/phase7/fusion_rp"],
    ["LC", "fusion", "R0+T2+K40", "runs/next_round/phase3/fusion"],
    ["B", "blend", "R0+T2+RP|runs/next_round/phase7/fusion_rp", str(THESIS / "results/next_round/phase7")],
    ["LCR", "fusion", "R0+T2+K40+RC", "runs/next_round/phase7/fusion_lcr"],
    ["LCRD", "fusion", "R0+T2+K40+RC+DF", "runs/next_round/phase7/fusion_lcrd"],
]
COMPARISONS = [("LRd", "L0"), ("LRf", "L0"), ("LRd", "LRf"), ("LRd", "L"), ("LC", "L"), ("LC", "L0"), ("B", "L"), ("B", "LC"),
               ("LCR", "LC"), ("LCR", "L"), ("LCR", "LRd"), ("LCR", "B"), ("LCRD", "LCR"), ("LCRD", "L")]
SEEDS = [1, 2, 3, 4, 5, 6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="first N frames only (smoke test)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda")
    d = selector.load_split(args.split, f"{bs.P1}/cells_in", f"{bs.P1}/cand_in/3d", DETS)
    if args.limit:
        d = selector.subset(d, np.arange(args.limit))
    items = d["items"]
    matches = {}
    for label, kind, name, weights in MODELS:
        for seed in ([0] if kind == "second" else SEEDS):
            if kind == "second":
                sc = [it["scores"] for it in items]
            elif kind == "fusion":
                sc = bs.fusion_scores(name, weights, seed, d, device)
            else:
                radar_expert, radar_weights = name.split("|")
                sc = bs.selector_scores(radar_expert, radar_weights, weights, seed, d, device, kind)
            matches[(label, seed)] = bs.frame_matches(bs.kept_frames(items, sc))
        print(f"scored {label} ({time.time() - t0:.0f} s)", flush=True)

    def value(label, w, b):
        return float(np.mean([bs.weighted_apcd(matches[k][b], w) for k in matches if k[0] == label]))

    scenes = sorted({it["seq"] for it in items})
    recordings = sorted({splits.recording(s) for s in scenes})
    frame_scene = np.array([scenes.index(it["seq"]) for it in items])
    frame_rec = np.array([recordings.index(splits.recording(it["seq"])) for it in items])
    ones = np.ones(len(items))
    point = {label: {b: value(label, ones, b) for b in bs.BINS} for label, *_ in MODELS}
    out = dict(split=args.split, frames=len(items), resamples=args.resamples, rng="numpy RandomState(0), one randint draw per resample",
               scenes=len(scenes), recordings={r: sum(splits.recording(s) == r for s in scenes) for r in recordings},
               point=point, scene={}, recording={})
    for level, unit in (("scene", frame_scene), ("recording", frame_rec)):
        n = int(unit.max()) + 1
        rng = np.random.RandomState(0)
        draws = {c: {b: [] for b in bs.BINS} for c in COMPARISONS}
        for _ in range(args.resamples):
            w = np.bincount(rng.randint(0, n, n), minlength=n)[unit].astype(np.float64)
            cache = {}
            for a, b2 in COMPARISONS:
                for b in bs.BINS:
                    for lab in (a, b2):
                        if (lab, b) not in cache:
                            cache[(lab, b)] = value(lab, w, b)
                    draws[(a, b2)][b].append(cache[(a, b)] - cache[(b2, b)])
        for (a, b2), per in draws.items():
            out[level][f"{a} - {b2}"] = {b: dict(diff=point[a][b] - point[b2][b], lo=float(np.nanpercentile(v, 2.5)),
                                                 hi=float(np.nanpercentile(v, 97.5)), p_le_0=float(np.mean(np.array(v) <= 0)),
                                                 nan_draws=int(np.isnan(v).sum()))
                                         for b, v in per.items()}
        print(f"{level} level done ({time.time() - t0:.0f} s)", flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    for c in out["recording"]:
        print(c, {b: f"{v['diff']:+.4f} scene [{out['scene'][c][b]['lo']:+.4f}, {out['scene'][c][b]['hi']:+.4f}] "
                     f"rec [{v['lo']:+.4f}, {v['hi']:+.4f}]" for b, v in out["recording"][c].items()}, flush=True)
    print("wrote", args.out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
