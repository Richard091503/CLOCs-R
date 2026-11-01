"""Select the radar gate (delta0, kappa of Eq. 2) on held-out training scenes, by center-distance AP beyond 50 m (Section II-C/E).

Trains the proposed support and Doppler set on the training scenes for each gate, and never reads the validation split.
The gate with the highest mean over seeds wins. Usage: python -m clocs_r.select_gate --out results/clocs_radar/gate_selection.json
"""
import argparse
import json
import multiprocessing as mp
import pathlib

import numpy as np
import torch

from clocs_r import cache_cells, fusion, metrics, radar_features, train_fusion

FAR = (50, 200)  # "beyond 50 m": the 50-100 m and 100-200 m bins scored together
DATA = {}


def score(net, names_idx, device):
    """Selection metric plus the per-bin results, on holdout."""
    base, radar = DATA["holdout"], DATA["holdout_radar"]
    records = [(it["gt"], b, s) for it, b, s in train_fusion.predict(net, base, radar, names_idx, device)]
    far = metrics.evaluate(records, bins=[FAR])[f"{FAR[0]}-{FAR[1]}"]
    bins = metrics.evaluate(records)
    return dict(APCD_far=far["APCD"], APBEV_far=far["APBEV@0.3"], n_gt_far=far["n_gt"],
                bins={k: {m: v[m] for m in ("n_gt", "APCD", "APBEV@0.3")} for k, v in bins.items()})


def run_seed(job):
    """In a forked child, so that CUDA is first used here and the parent's cached frames are shared."""
    feature_set, seed, epochs = job
    torch.set_num_threads(1)
    device = torch.device("cuda")
    names_idx = [radar_features.FEATURES.index(f) for f in radar_features.FEATURE_SETS[feature_set]]
    net, log = train_fusion.fit(DATA["train"], DATA["train_radar"], names_idx, epochs, seed, device, f"{feature_set} s{seed}")
    return dict(seed=seed, final_loss=log[-1]["loss"], **score(net, names_idx, device))


def baseline(job):
    """SECOND alone and the trained R0 networks, through the same scoring path."""
    path = job
    torch.set_num_threads(1)
    device = torch.device("cuda")
    if path is None:
        return score(None, [], device)
    net = fusion.FusionNet(0).to(device)
    net.load_state_dict(torch.load(path, map_location=device)["model"])
    return score(net, [], device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", default="runs/cells")
    ap.add_argument("--cand3d", default="runs/candidates3d")
    ap.add_argument("--cand2d", default="runs/candidates2d")
    ap.add_argument("--feature_set", default="R3")
    ap.add_argument("--delta0", default="0.25,0.5,1.0")
    ap.add_argument("--kappa", default="0.01,0.02,0.04")
    ap.add_argument("--sweeps", type=int, default=cache_cells.SWEEPS)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    gates = [(float(d), float(k)) for d in args.delta0.split(",") for k in args.kappa.split(",")]
    results = dict(args=vars(args), selection_metric="APCD beyond 50 m on holdout, mean over seeds", gates=[])
    out = pathlib.Path(args.out)
    ctx = mp.get_context("fork")

    DATA["train"], DATA["train_radar"] = cache_cells.load("train", args.cells, 0, 0, 0, False)
    DATA["holdout"], DATA["holdout_radar"] = cache_cells.load("holdout", args.cells, 0, 0, 0, False)
    with ctx.Pool(1) as pool:
        results["second"] = pool.map(baseline, [None])[0]
    with ctx.Pool(3) as pool:
        results["R0"] = pool.map(baseline, [f"runs/fusion/R0_s{s}/fusion.pth" for s in seeds])
    print("SECOND", round(results["second"]["APCD_far"], 4), "R0", [round(r["APCD_far"], 4) for r in results["R0"]], flush=True)

    for delta0, kappa in gates:
        for split in ("train", "holdout"):
            cache_cells.build(split, args.cells, args.cand3d, args.cand2d, delta0, kappa, args.sweeps)
        DATA["train_radar"] = cache_cells.load_radar("train", args.cells, delta0, kappa, args.sweeps)
        DATA["holdout_radar"] = cache_cells.load_radar("holdout", args.cells, delta0, kappa, args.sweeps)
        with ctx.Pool(len(seeds)) as pool:
            runs = pool.map(run_seed, [(args.feature_set, s, args.epochs) for s in seeds])
        far = [r["APCD_far"] for r in runs]
        results["gates"].append(dict(delta0=delta0, kappa=kappa, APCD_far_mean=float(np.mean(far)), APCD_far_std=float(np.std(far)), runs=runs))
        print(f"delta0 {delta0} kappa {kappa}: APCD beyond 50 m {np.mean(far):.4f} (seeds {[round(x, 4) for x in far]})", flush=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=1)

    best = max(results["gates"], key=lambda g: g["APCD_far_mean"])
    results["selected"] = dict(delta0=best["delta0"], kappa=best["kappa"], APCD_far_mean=best["APCD_far_mean"])
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("selected", results["selected"], flush=True)


if __name__ == "__main__":
    main()
