"""Apply the selection rule of results/full_dataset/PLAN.md to holdout results, without reading val.

Usage: python -m clocs_r.select_full --selection results/full_dataset/selection.json --out results/full_dataset/selected.json
"""
import argparse
import json

import numpy as np

CANDIDATES = ("R2", "R3", "R2+T2", "R3+T2")  # the CLOCs-R configurations competing for selection
METRIC = ("overall", "50-200", "APCD")  # center-distance AP beyond 50 m


def score(entry):
    group, bin_, metric = METRIC
    values = [run["holdout"][group][bin_][metric] for run in entry["runs"]]
    return float(np.mean(values)), [float(v) for v in values]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/full_dataset/selection.json")
    ap.add_argument("--out", default="results/full_dataset/selected.json")
    args = ap.parse_args()
    results = json.load(open(args.selection))
    assert all("val" not in run for entry in results.values() for run in entry["runs"]), "selection results must not contain val"
    table = {name: score(results[name]) for name in results}
    winner = max(CANDIDATES, key=lambda name: table[name][0])
    twin = "R0+T2" if "T2" in winner.split("+") else "R0"
    out = dict(rule="highest mean holdout center-distance AP beyond 50 m among " + ", ".join(CANDIDATES),
               winner=winner, twin=twin, holdout={name: dict(mean=m, seeds=v) for name, (m, v) in table.items()})
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for name, (m, v) in sorted(table.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:8s} {m:.4f}  seeds {[round(x, 4) for x in v]}" + ("  <- selected" if name == winner else ""))
    print("radar-free twin:", twin)


if __name__ == "__main__":
    main()
