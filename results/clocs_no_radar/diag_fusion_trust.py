"""Why does CLOCs without radar lose to SECOND on aiMotive at range? One-off check, from the repo root.

1. How reliable is a confident LiDAR candidate with no 2D match, on each split?
2. Does fusion improve if trained on scenes the detectors never saw? Two-fold by scene within holdout, which fusion
   can be judged on without touching val, and then a model trained on all of holdout, scored on val.

Usage: python results/clocs_no_radar/diag_fusion_trust.py
"""
import json
import pathlib

import numpy as np
import torch

from clocs_r import cache_cells, metrics, train_fusion

OUT = pathlib.Path(__file__).parent
BINS = {"0-50": (0, 50), "50-100": (50, 100), "100-200": (100, 200)}
SEEDS = (1, 2, 3)
device = torch.device("cuda")


def reliability(base):
    """Among scoreable candidates with s3D >= 0.5: share that are true cars (BEV IoU >= 0.7), with and without a 2D match."""
    out = {}
    for name, (lo, hi) in BINS.items():
        n = {True: 0, False: 0}
        pos = {True: 0, False: 0}
        for it in base:
            matched = np.zeros(len(it["scores"]), bool)
            matched[it["column"][it["pairs"][:, 0] > 0]] = True
            rng = it["dist"] * cache_cells.MAX_RANGE
            mask = (it["scores"] >= 0.5) & it["keep_eval"] & (rng >= lo) & (rng < hi) & (it["weight"] > 0)
            for m in (True, False):
                sel = mask & (matched == m)
                n[m] += int(sel.sum())
                pos[m] += int(it["target"][sel].sum())
        out[name] = {"matched": {"n": n[True], "true_car": pos[True] / max(n[True], 1)},
                     "unmatched": {"n": n[False], "true_car": pos[False] / max(n[False], 1)},
                     "unmatched_share_of_true": pos[False] / max(pos[True] + pos[False], 1)}
    return out


def score(net, items):
    records = [(it["gt"], b, s) for it, b, s in train_fusion.predict(net, items, [None] * len(items), [], device)]
    return metrics.evaluate(records)


def summary(res):
    return {b: {m: round(res[b][m], 3) for m in ("AP3D@0.5", "AP3D@0.7", "APBEV@0.3", "APCD", "recall@BEV0.3")} for b in res}


def mean_results(runs):
    return {b: {m: float(np.mean([r[b][m] for r in runs])) for m in runs[0][b]} for b in runs[0]}


def main():
    results = {"reliability": {}}
    data = {}
    for split in ("train", "holdout", "val"):
        data[split], _ = cache_cells.load(split, "runs/cells", 0.5, 0.02, 3, use_radar=False)
        results["reliability"][split] = reliability(data[split])
        print(split, json.dumps(results["reliability"][split]), flush=True)
    del data["train"]

    holdout = data["holdout"]
    scenes = sorted({it["seq"] for it in holdout})
    folds = [set(scenes[0::2]), set(scenes[1::2])]
    print("SECOND holdout", summary(score(None, holdout)), flush=True)
    runs = []
    for seed in SEEDS:
        records = []
        for fold in folds:
            train = [it for it in holdout if it["seq"] not in fold]
            test = [it for it in holdout if it["seq"] in fold]
            net, _ = train_fusion.fit(train, [None] * len(train), [], 3, seed, device)
            records += [(it["gt"], b, s) for it, b, s in train_fusion.predict(net, test, [None] * len(test), [], device)]
        runs.append(metrics.evaluate(records))
    results["holdout_trained_cross_fit_on_holdout"] = mean_results(runs)
    print("CLOCs trained on the other holdout scenes, scored on holdout", summary(results["holdout_trained_cross_fit_on_holdout"]), flush=True)

    runs = []
    for seed in SEEDS:
        net, _ = train_fusion.fit(holdout, [None] * len(holdout), [], 3, seed, device)
        runs.append(score(net, data["val"]))
    results["holdout_trained_on_val"] = mean_results(runs)
    print("CLOCs trained on all holdout, scored on val", summary(results["holdout_trained_on_val"]), flush=True)
    with open(OUT / "diag_fusion_trust.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
