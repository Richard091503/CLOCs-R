"""Collect the Table II runs, select the final feature set on holdout (Section II-C), and test it with the camera removed (Section II-E).

The set with the highest mean center-distance AP beyond 50 m on holdout is selected. Val is reported for every set and plays no part in that.

Usage, from the repo root, after all runs: python results/clocs_radar/ablation/make_tables.py
"""
import json
import pathlib
import shutil

import numpy as np
import torch

from clocs_r import aimotive, cache_cells, fusion, metrics, radar_features, train_fusion

OUT = pathlib.Path(__file__).parent
SETS = tuple(radar_features.FEATURE_SETS)
SEEDS = (1, 2, 3)
GATE = (0.5, 0.02, 3)
SPLITS = ("holdout", "val")
FAR = f"{train_fusion.FAR[0]}-{train_fusion.FAR[1]}"
METRICS = ("n_gt", "APBEV@0.3", "APCD", "AP3D@0.5", "AP3D@0.7", "recall@BEV0.3")
device = torch.device("cuda")


def load_net(name, seed):
    idx = [radar_features.FEATURES.index(f) for f in radar_features.FEATURE_SETS[name]]
    net = fusion.FusionNet(len(idx)).to(device)
    net.load_state_dict(torch.load(f"runs/fusion/{name}_s{seed}/fusion.pth", map_location=device)["model"])
    return net, idx


def far_scores(net, base, radar, idx, nocam=False):
    records = [(it["gt"], b, s) for it, b, s in train_fusion.predict(net, base, radar, idx, device, nocam)]
    return metrics.evaluate(records, bins=aimotive.DIST_BINS + [train_fusion.FAR])


def summarize(runs):
    """{bin: {metric: mean, metric_min, metric_max}} over seeds."""
    out = {}
    for b in runs[0]:
        out[b] = {}
        for m in METRICS:
            v = [r[b][m] for r in runs]
            out[b][m] = float(np.mean(v))
            if m != "n_gt":
                out[b][m + "_min"], out[b][m + "_max"] = float(np.min(v)), float(np.max(v))
    return out


def main():
    data = {s: cache_cells.load(s, "runs/cells", *GATE) for s in SPLITS}
    results = dict(gate=dict(delta0=GATE[0], kappa=GATE[1], sweeps=GATE[2]), second={}, sets={})
    for split in SPLITS:
        base, radar = data[split]
        results["second"][split] = {k: {m: v[m] for m in METRICS} for k, v in far_scores(None, base, radar, []).items()}
    for name in SETS:
        results["sets"][name] = dict(features=radar_features.FEATURE_SETS[name])
        runs = [json.load(open(f"runs/fusion/{name}_s{s}/results.json")) for s in SEEDS]
        results["sets"][name]["params"] = runs[0]["params"]
        results["sets"][name]["final_loss"] = [r["loss"][-1]["loss"] for r in runs]
        for split in SPLITS:
            base, radar = data[split]
            if FAR not in runs[0][split]["overall"]:  # R0 and R3 ran before this bin was part of training results
                for s, r in zip(SEEDS, runs):
                    net, idx = load_net(name, s)
                    r[split]["overall"] = far_scores(net, base, radar, idx)
            results["sets"][name][split] = summarize([r[split]["overall"] for r in runs])
            results["sets"][name][split + "_conditions"] = {c: summarize([r[split][c] for r in runs]) for c in ("highway", "urban", "no_tunnels")}
        for s, r in zip(SEEDS, runs):
            with open(OUT / f"{name}_seed{s}.json", "w") as f:
                json.dump(r, f, indent=1)
            shutil.copy(f"runs/fusion/{name}_s{s}/fusion.pth", OUT / f"{name}_seed{s}.pth")
        print(name, {s: round(results["sets"][name][s][FAR]["APCD"], 4) for s in SPLITS}, flush=True)

    selected = max(SETS, key=lambda n: results["sets"][n]["holdout"][FAR]["APCD"])
    results["selected"] = selected
    print("selected", selected, flush=True)

    # Camera removed: every candidate uses the reserved row. R0 shows what is left without radar as well.
    results["camera_removed"] = {}
    for name in dict.fromkeys(("R0", selected)):
        results["camera_removed"][name] = {}
        for split in SPLITS:
            base, radar = data[split]
            runs = []
            for s in SEEDS:
                net, idx = load_net(name, s)
                runs.append(far_scores(net, base, radar, idx, nocam=True))
            results["camera_removed"][name][split] = summarize(runs)
        print("camera removed", name, {s: round(results["camera_removed"][name][s][FAR]["APCD"], 4) for s in SPLITS}, flush=True)

    with open(OUT / "summary.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
