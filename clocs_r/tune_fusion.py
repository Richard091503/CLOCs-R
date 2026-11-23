"""Train (or re-score) CLOCs fusion networks on the cached cells of SECOND's candidates, several seeds in parallel.

A configuration joins tokens with "+". Every configuration here starts with R0+T2:
    R0    no radar point features (the cells carry only CLOCs' own channels)
    T2    residual output: the network starts exactly at SECOND's logit and learns a correction
and adds the second sensor:
    K40   camera CLOCs: Cascade R-CNN pairs, each candidate keeping only 2D boxes at least as strong as the frame's
          40th-highest 2D score (visibility.TOP_2D)                                         R0+T2+K40        = LC
    RP    radar CLOCs: the radar detector D2's boxes take the place of the 2D boxes as CLOCs' pairs (radar_pairs.py)
                                                                                            R0+T2+RP         = LRd
    RC    with K40: one network over SECOND, Cascade R-CNN and D2, camera and radar cells side by side
          (radar_pairs.combined_cells)                                                      R0+T2+K40+RC     = LCR
    DF    with RC: in training only, camera cells of candidates beyond DF_M hidden with probability DF_P
          (fusion.drop_far_camera)                                                          R0+T2+K40+RC+DF  = LCR-D

Other configurations from the project's development (radar point features, gating, radar candidates, ...) are documented
in docs/MODELS.md; their code is on the `thesis` branch.

Usage:
    python -m clocs_r.tune_fusion --configs R0+T2+K40+RC --splits select --seeds 1,2,3 --radar_dets runs/lcr/d2_dets \
        --cells runs/lcr/cells --cand3d runs/lcr/cand/3d --cand2d runs/lcr/cand/2d --weights runs/lcr/fusion \
        --out results/lcr/lcr_select.json
    add --from_weights to re-score saved networks (e.g. on val) instead of training them.
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import time

import numpy as np
import torch

from clocs_r import cache_cells, fusion, radar_candidates, radar_pairs, train_fusion, visibility

GATE = (0.5, 0.02)  # radar gate (delta0, kappa) of the cached cells; part of the cache names and of the saved settings
SEEDS = (1, 2, 3)
DATA = {}
STAGGER_S = 20  # s between the CUDA start-ups of parallel seeds: simultaneous start-ups twice failed on a shared GPU
TOKENS = ("R0", "T2", "K40", "RP", "RC", "DF")
DF_M, DF_P = 50.0, 0.5
# Settings saved with every network, and checked when it is loaded. These values are fixed here (the radar point-feature
# options of earlier configurations are off), so the settings of saved networks still match.
LEGACY = dict(radar=False, base="R2", far_targets=False, sweeps=cache_cells.SWEEPS, motion=False, rcs=False)
P3_DEFAULTS = dict(vis=[], min_iou=0.0, top2d=0, cam_dropout=0.0, oversample_m=0.0, oversample=2.0)


def parse(config):
    """Turn a name such as R0+T2+K40+RC into its settings."""
    parts = config.split("+")
    assert set(parts) <= set(TOKENS) and {"R0", "T2"} <= set(parts), f"{config}: unknown token, or no R0+T2"
    cfg = dict(LEGACY, net=dict(residual=True, logit_input=False))
    if "K40" in parts:
        cfg["p3"] = dict(P3_DEFAULTS, top2d=visibility.TOP_2D)
    if "RP" in parts:
        assert "K40" not in parts and "RC" not in parts, f"{config}: RP replaces the camera pairs"
        cfg["rp"] = dict(min_score=radar_candidates.RD_MIN, match_m=radar_candidates.MATCH_M)
    if "RC" in parts:
        assert "K40" in parts, f"{config}: RC adds radar-detector pairs to R0+T2+K40"
        cfg["rc"] = dict(min_score=radar_candidates.RD_MIN, match_m=radar_candidates.MATCH_M)
    if "DF" in parts:
        assert "RC" in parts, f"{config}: DF hides camera cells beside the radar-detector cells of RC"
        cfg["far_cam_dropout"] = dict(far_m=DF_M, p=DF_P)
    return cfg


def n_extra(cfg):
    """Input channels after [IoU, 2D score, s3D, distance]: the two radar-detector places of token RC."""
    return 2 * bool(cfg.get("rc"))


def make_net(cfg):
    return fusion.FusionNet(n_extra(cfg), **cfg["net"])


def pairing(cfg, top2d):
    """Per-frame (min_iou, s2d_floor) of LC's K40 floor; None otherwise (RC applies the floor when it builds its cells)."""
    if cfg.get("p3") and not cfg.get("rc"):
        return [(0.0, t) for t in top2d]
    return None


def prepare(split, key, cells, cand3d, cand2d, radar_dets, kind):
    """Load a split's cells into DATA[key], with the K40 floors and the pairs of the configuration's kind (base, rp, rc)."""
    DATA[key], _ = cache_cells.load(split, cells, *GATE, cache_cells.SWEEPS, use_radar=False)
    visibility.build(split, cells, cand3d, cand2d)  # cached beside the cells; only the top-40 2D score is used here
    DATA["top2d_" + key] = visibility.load(split, cells, DATA[key])["top2d"]
    if kind == "rp":
        DATA[key] = radar_pairs.replace_pairs(DATA[key], radar_pairs.build(split, cells, radar_dets, DATA[key]))
    elif kind == "rc":
        DATA[key] = radar_pairs.add_pairs(DATA[key], radar_pairs.build(split, cells, radar_dets, DATA[key]), DATA["top2d_" + key])


def kind(cfg):
    return "rc" if cfg.get("rc") else "rp" if cfg.get("rp") else "base"


def load_net(cfg, name, weights, seed, far_weight, device):
    path = pathlib.Path(weights) / f"{name}_s{seed}.pth"
    state = torch.load(path, map_location="cpu")
    assert state["settings"] == cfg and state.get("far_weight", 1.0) == far_weight and tuple(state.get("gate", GATE)) == tuple(GATE), \
        f"{path} was trained differently"
    net = make_net(cfg).to(device)
    net.load_state_dict(state["model"])
    return net


def run_seed(job):
    """In a forked child, so that CUDA is first used here and the parent's cached frames are shared."""
    name, cfg, seed, epochs, eval_splits, out, far_weight, from_weights = job
    time.sleep(STAGGER_S * (seed - 1) % (3 * STAGGER_S))  # start CUDA in the forked children one at a time (see STAGGER_S)
    torch.set_num_threads(1)
    device = torch.device("cuda")
    path = pathlib.Path(out) / f"{name}_s{seed}.pth"
    if from_weights:  # score the saved networks again, e.g. on val after selection, instead of retraining them
        net = load_net(cfg, name, out, seed, far_weight, device)
        result = dict(seed=seed, weights=str(path))
    else:
        dfc = cfg.get("far_cam_dropout")
        net, log = train_fusion.fit(DATA["train"], epochs, seed, device, f"{name} s{seed}", net_kwargs=cfg["net"],
                                    far_weight=far_weight, n_extra=n_extra(cfg), pairing=pairing(cfg, DATA["top2d_train"]),
                                    far_cam_dropout=(dfc["far_m"] / cache_cells.MAX_RANGE, dfc["p"]) if dfc else None)
        torch.save(dict(model=net.state_dict(), config=name, settings=cfg, gate=GATE, far_weight=far_weight), path)
        result = dict(seed=seed, final_loss=log[-1]["loss"] if log else None, loss=log)
    for split in eval_splits:
        result[split] = train_fusion.evaluate(net, DATA[split], device, pairing(cfg, DATA["top2d_" + split]))
    result["params"] = sum(p.numel() for p in net.parameters())
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="comma-separated; all of one kind (K40, RP or RC), as they share the data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="select")
    ap.add_argument("--weights", default="runs/lcr/fusion")
    ap.add_argument("--cells", default="runs/lcr/cells")
    ap.add_argument("--cand3d", default="runs/lcr/cand/3d")
    ap.add_argument("--cand2d", default="runs/lcr/cand/2d")
    ap.add_argument("--radar_dets", default="runs/lcr/d2_dets", help="exported D2 boxes, for RP and RC configurations")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--far_weight", type=float, default=2.0, help="range-weighted fusion loss (range_weight.py)")
    ap.add_argument("--from_weights", action="store_true", help="evaluate the networks saved in --weights instead of training")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)), help="comma-separated training seeds, one process each")
    ap.add_argument("--train_split", default="fit", help="clocs_r.splits name of the fusion training frames")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    eval_splits = args.splits.split(",")
    names = args.configs.split(",")
    kinds = {kind(parse(n)) for n in names}
    assert len(kinds) == 1, "K40, RP and RC configurations pair the candidates differently; run them separately"
    pathlib.Path(args.weights).mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(args.out)
    results = json.load(open(out)) if out.exists() else {}
    for split, key in [(s, s) for s in eval_splits] + ([] if args.from_weights else [(args.train_split, "train")]):
        prepare(split, key, args.cells, args.cand3d, args.cand2d, args.radar_dets, next(iter(kinds)))
    ctx = mp.get_context("fork")
    for name in names:
        cfg = parse(name)
        with ctx.Pool(len(seeds)) as pool:
            runs = pool.map(run_seed, [(name, cfg, s, args.epochs, eval_splits, args.weights, args.far_weight, args.from_weights)
                                       for s in seeds])
        prev = results.get(name)
        if prev and prev.get("settings") == cfg and prev.get("far_weight") == args.far_weight and prev.get("gate") == list(GATE):
            # another seed group of the same run: keep both, by seed
            runs = sorted({**{r["seed"]: r for r in prev["runs"]}, **{r["seed"]: r for r in runs}}.values(), key=lambda r: r["seed"])
        results[name] = dict(settings=cfg, far_weight=args.far_weight, gate=list(GATE), nocam=False, runs=runs)
        far = {split: [r[split]["overall"]["50-200"]["APCD"] for r in runs] for split in eval_splits}
        print(name, {k: f"{np.mean(v):.4f} ({[round(x, 4) for x in v]})" for k, v in far.items()}, flush=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
