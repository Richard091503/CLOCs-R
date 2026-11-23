"""Range switch (results/next_round/PLAN.md, Phase 2): a saved fusion network's scores within --switch_m metres and
SECOND's own scores beyond, with no training. Beyond the switch it is SECOND by construction.

Usage: python -m clocs_r.range_switch --configs R2+T2,R0+T2 --weights runs/next_round/phase1/fusion_insample \
           --cells runs/next_round/phase1/cells_in --splits select,val --seeds 1,2,3,4,5,6 --out results/next_round/phase2/switch.json
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from clocs_r import aimotive, cache_cells, fusion, metrics, train_fusion, tune_fusion


def switch_scores(boxes, fused, lidar, switch_m):
    """Fused scores for candidates under switch_m metres of BEV range, the LiDAR detector's scores beyond."""
    return np.where(np.hypot(boxes[:, 0], boxes[:, 1]) < switch_m, fused, lidar)


@torch.no_grad()
def evaluate(net, base, radar, idx, device, switch_m, nms_thresh=0.1):
    """As train_fusion.evaluate, with the range switch applied before NMS. Also counts candidates whose NMS outcome
    differs from SECOND alone beyond the switch (kept by one and not the other)."""
    net.eval()
    records, differ = [], 0
    for i, item in enumerate(base):
        x, col = train_fusion.frame_cells(item, radar[i], idx, device)
        fused = torch.sigmoid(net(x, col, len(item["scores"]))).cpu().numpy()
        scores = switch_scores(item["boxes"], fused, item["scores"], switch_m)
        mask = item["keep_eval"]
        boxes, s, s3d = item["boxes"][mask], scores[mask], item["scores"][mask]
        keep = metrics.nms_bev(boxes, s, nms_thresh)
        far = np.hypot(boxes[:, 0], boxes[:, 1]) >= switch_m
        mine, alone = np.zeros(len(s), bool), np.zeros(len(s), bool)
        mine[keep], alone[metrics.nms_bev(boxes, s3d, nms_thresh)] = True, True
        differ += int((far & (mine != alone)).sum())
        records.append((item["gt"], boxes[keep], s[keep], item["odd"], item["seq"]))
    out = {name: metrics.evaluate([r[:3] for r in rows], bins=aimotive.DIST_BINS + [train_fusion.FAR])
           for name, rows in train_fusion.condition_groups(records).items()}
    out["far_nms_differences"] = differ
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="R2+T2,R0+T2")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--cand3d", default="runs/candidates3d")
    ap.add_argument("--cand2d", default="runs/candidates2d")
    ap.add_argument("--splits", default="select,val")
    ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--switch_m", type=float, default=50.0)
    ap.add_argument("--gate", default="0.5,0.02")
    ap.add_argument("--far_weight", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tune_fusion.GATE = tuple(float(x) for x in args.gate.split(","))
    device = torch.device("cuda")
    splits_ = args.splits.split(",")
    data = {s: cache_cells.load(s, args.cells, *tune_fusion.GATE, cache_cells.SWEEPS, use_radar=False)[0] for s in splits_}
    results = {}
    for name in args.configs.split(","):
        cfg = tune_fusion.parse(name)
        idx = tune_fusion.names_idx(cfg)
        radar = {s: tune_fusion.radar_data(s, cfg, args) if idx else [None] * len(data[s]) for s in splits_}
        runs = []
        for seed in map(int, args.seeds.split(",")):
            path = pathlib.Path(args.weights) / f"{name}_s{seed}.pth"
            state = torch.load(path, map_location="cpu")
            assert state["settings"] == cfg and state.get("far_weight", 1.0) == args.far_weight, f"{path} was trained differently"
            net = fusion.FusionNet(len(idx), **cfg["net"]).to(device)
            net.load_state_dict(state["model"])
            r = dict(seed=seed, weights=str(path), switch_m=args.switch_m)
            for s in splits_:
                r[s] = evaluate(net, data[s], radar[s], idx, device, args.switch_m)
            runs.append(r)
            print(name, seed, {s: round(r[s]["overall"]["50-200"]["APCD"], 4) for s in splits_},
                  {s: r[s]["far_nms_differences"] for s in splits_}, flush=True)
        results[f"switch({name})"] = dict(settings=cfg, switch_m=args.switch_m, runs=runs)
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
