"""Baseline: SECOND's original scores with standard NMS, no fusion. Usage: python -m clocs_r.baseline_second"""
import json
import pathlib
import sys

from clocs_r import cache_cells
from clocs_r.train_fusion import evaluate


def main(cells="runs/cells", out="runs/fusion/second_only", delta0=0.5, kappa=0.02):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in ("holdout", "val"):
        base, radar = cache_cells.load(split, cells, delta0, kappa, cache_cells.SWEEPS, use_radar=False)
        results[split] = evaluate(None, base, radar, [], None)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    print({k: {m: round(v[m], 3) for m in ("n_gt", "AP3D@0.5", "APBEV@0.3", "APCD", "recall@BEV0.3")} for k, v in results["val"]["overall"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
