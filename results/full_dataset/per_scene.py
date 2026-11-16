"""Per-scene scores of SECOND, the selected CLOCs-R and its radar-free twin, for the scene-level radar gain in PLAN.md.

Run after clocs_r/run_full.sh (it reads selected.json and the saved fusion networks):
    python results/full_dataset/per_scene.py [--split val] [--models R3+T2,R2+T2,R0+T2 --seeds 1,2,3,4,5,6 --suffix _6seeds]
Writes per_scene_<split>.json: for each scene its condition, number of cars beyond 50 m, and center-distance AP / BEV AP@0.3
beyond 50 m and overall for SECOND and for each seed of both fusion models.
"""
import argparse
import collections
import json
import pathlib

import torch

from clocs_r import aimotive, cache_cells, fusion, metrics, train_fusion, tune_fusion

HERE = pathlib.Path(__file__).parent
BINS = aimotive.DIST_BINS + [train_fusion.FAR]


def by_scene(predictions):
    scenes = collections.defaultdict(list)
    odd = {}
    for item, boxes, scores in predictions:
        scenes[item["seq"]].append((item["gt"], boxes, scores))
        odd[item["seq"]] = item["odd"]
    return {seq: dict(odd=odd[seq], metrics=metrics.evaluate(rows, bins=BINS)) for seq, rows in scenes.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--cells", default="runs/cells")
    ap.add_argument("--weights", default="runs/fusion_full")
    ap.add_argument("--selected", default=str(HERE / "selected.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", default=None, help="comma-separated configurations (default: the selected model and its twin)")
    ap.add_argument("--seeds", default=",".join(map(str, tune_fusion.SEEDS)))
    ap.add_argument("--suffix", default="", help="appended to the output file name")
    ap.add_argument("--gate", default=",".join(map(str, tune_fusion.GATE)), help="radar gate the networks were trained with")
    ap.add_argument("--cand3d", default="runs/candidates3d")
    ap.add_argument("--cand2d", default="runs/candidates2d")
    args = ap.parse_args()
    tune_fusion.GATE = tuple(float(x) for x in args.gate.split(","))
    selected = json.load(open(args.selected))
    device = torch.device("cuda")
    base, _ = cache_cells.load(args.split, args.cells, *tune_fusion.GATE, cache_cells.SWEEPS, use_radar=False)
    out = {"SECOND": by_scene(train_fusion.predict(None, base, None, [], None))}
    for name in (args.models.split(",") if args.models else (selected["winner"], selected["twin"])):
        cfg = tune_fusion.parse(name)
        idx = tune_fusion.names_idx(cfg)
        radar = tune_fusion.radar_data(args.split, cfg, args)  # this configuration's sweeps, motion correction and RCS
        out[name] = {}
        for seed in map(int, args.seeds.split(",")):
            net = fusion.FusionNet(len(idx), **cfg["net"]).to(device)
            net.load_state_dict(torch.load(pathlib.Path(args.weights) / f"{name}_s{seed}.pth", map_location="cpu")["model"])
            out[name][seed] = by_scene(train_fusion.predict(net, base, radar if idx else [None] * len(base), idx, device))
    path = pathlib.Path(args.out or HERE / f"per_scene_{args.split}{args.suffix}.json")
    with open(path, "w") as f:
        json.dump(dict(selected=selected, scenes=out), f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
