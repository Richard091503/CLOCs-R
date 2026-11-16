"""Train and evaluate the CLOCs-R fusion network for one radar feature set, from cached cells.

Usage: python -m clocs_r.train_fusion --feature_set R3 --out runs/fusion/R3_s3
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from clocs_r import aimotive, cache_cells, fusion, metrics, radar_features, range_weight, splits

FAR = (50, 200)  # the 50-100 m and 100-200 m bins scored together, for the selection metric of Section II-C


def frame_cells(item, radar, names_idx, device, nocam=False):
    """Compose the (P, 4+m) input of one frame: base cells plus the candidate's radar features in each cell."""
    if nocam:
        column = np.arange(len(item["scores"]))
        pairs = np.full((len(column), 2), fusion.MISSING, dtype=np.float32)
    else:
        column, pairs = item["column"], item["pairs"]
    cells = np.concatenate([pairs.astype(np.float32), item["scores"][column, None], item["dist"][column, None]], 1)
    x = torch.from_numpy(cells).to(device)
    col = torch.from_numpy(np.asarray(column, dtype=np.int64)).to(device)
    if names_idx:
        x = torch.cat([x, torch.from_numpy(np.asarray(radar[:, names_idx], dtype=np.float32)).to(device)[col]], 1)
    return x, col


@torch.no_grad()
def predict(net, base, radar, names_idx, device, nocam=False, nms_thresh=0.1, score_fn=None):
    """Yields (item, kept boxes, scores) after fused scoring (or score_fn) and NMS, ready for metrics.evaluate."""
    net_was_training = net.training if net is not None else False
    if net is not None:
        net.eval()
    for i, item in enumerate(base):
        if net is None:
            scores = item["scores"]
        else:
            x, col = frame_cells(item, radar[i], names_idx, device, nocam)
            scores = torch.sigmoid(net(x, col, len(item["scores"]))).cpu().numpy()
        mask = item["keep_eval"]
        boxes, s = item["boxes"][mask], scores[mask]
        keep = metrics.nms_bev(boxes, s, nms_thresh)
        yield item, boxes[keep], s[keep]
    if net_was_training:
        net.train()


def condition_groups(records, odd=lambda r: r[3], seq=lambda r: r[4]):
    """Named subsets of a split's records: all, clear and adverse weather, each driving condition, and no tunnels."""
    out = {"overall": records,
           "clear": [r for r in records if odd(r) in splits.CLEAR_ODDS],
           "adverse": [r for r in records if odd(r) in splits.ADVERSE_ODDS]}
    for o in splits.ODDS:
        out[o] = [r for r in records if odd(r) == o]
    out["no_tunnels"] = [r for r in records if seq(r) not in splits.TUNNEL_SCENES]
    return {name: rows for name, rows in out.items() if rows}


def evaluate(net, base, radar, names_idx, device, nocam=False):
    """Metrics by distance bin (with beyond 50 m as one more bin), overall and for each group of driving conditions."""
    records = [(it["gt"], b, s, it["odd"], it["seq"]) for it, b, s in predict(net, base, radar, names_idx, device, nocam)]
    return {name: metrics.evaluate([r[:3] for r in rows], bins=aimotive.DIST_BINS + [FAR])
            for name, rows in condition_groups(records).items()}


def fit(train, train_radar, names_idx, epochs, seed, device, label="", net_kwargs=None, far_weight=1.0):
    """Train a fusion network on cached frames. Returns the network and its loss log.

    far_weight: weight of candidates beyond 60 m in the loss, ramping up from 40 m (range_weight.py); 1 leaves it unweighted.
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    net = fusion.FusionNet(len(names_idx), **(net_kwargs or {})).to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    steps = epochs * len(train)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=3e-3, total_steps=steps, pct_start=0.4, div_factor=10,
                                                final_div_factor=1e4, cycle_momentum=True, base_momentum=0.85, max_momentum=0.95)
    log, t0, step, running = [], time.time(), 0, 0.0
    net.train()
    for epoch in range(epochs):
        for i in rng.permutation(len(train)):
            item = train[i]
            x, col = frame_cells(item, train_radar[i], names_idx, device)
            logits = net(x, col, len(item["scores"]))
            tgt = torch.from_numpy(item["target"]).to(device)
            w = item["weight"]
            if far_weight != 1.0:
                w = w * range_weight.weight(item["dist"] * cache_cells.MAX_RANGE, far_weight=far_weight).astype(np.float32)
            w = torch.from_numpy(w).to(device)
            loss = fusion.focal_loss(logits, tgt, w) / max(tgt.sum().item(), 1.0)  # normalized by positives, as in CLOCs
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            step += 1
            running += loss.item()
            if step % 200 == 0:
                log.append(dict(step=step, loss=running / 200, lr=sched.get_last_lr()[0]))
                running = 0.0
                if step % 2000 == 0:
                    print(f"{label} step {step}/{steps} loss {log[-1]['loss']:.4f} {step / (time.time() - t0):.0f} it/s", flush=True)
    return net, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--cells", default="runs/cells")
    ap.add_argument("--delta0", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=0.02)
    ap.add_argument("--sweeps", type=int, default=cache_cells.SWEEPS)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--eval_splits", default="holdout,val")
    ap.add_argument("--nocam", action="store_true", help="also evaluate with all 2D candidates removed")
    ap.add_argument("--residual", action="store_true", help="T2: the fused logit is SECOND's logit plus the network output")
    ap.add_argument("--far_weight", type=float, default=1.0, help="loss weight beyond 60 m (ramp from 40 m)")
    args = ap.parse_args()

    device = torch.device("cuda")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names_idx = [radar_features.FEATURES.index(f) for f in radar_features.FEATURE_SETS[args.feature_set]]

    use_radar = bool(names_idx)
    t0 = time.time()
    train, train_radar = cache_cells.load(args.train_split, args.cells, args.delta0, args.kappa, args.sweeps, use_radar)
    net, log = fit(train, train_radar, names_idx, args.epochs, args.seed, device, args.feature_set,
                   net_kwargs=dict(residual=args.residual), far_weight=args.far_weight)
    torch.save(dict(model=net.state_dict(), args=vars(args)), out / "fusion.pth")

    results = dict(args=vars(args), train_seconds=time.time() - t0, loss=log, params=sum(p.numel() for p in net.parameters()))
    del train, train_radar
    for split in args.eval_splits.split(","):
        base, radar = cache_cells.load(split, args.cells, args.delta0, args.kappa, args.sweeps, use_radar)
        results[split] = evaluate(net, base, radar, names_idx, device)
        if args.nocam:
            results[split + "_nocam"] = evaluate(net, base, radar, names_idx, device, nocam=True)
        if split == "val":
            results["latency_ms"] = measure_latency(net, base, radar, names_idx, device)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    ov = results.get("val", results[args.eval_splits.split(",")[0]])["overall"]
    print(args.feature_set, {k: {m: round(v[m], 3) for m in ("AP3D@0.5", "APBEV@0.3", "APCD")} for k, v in ov.items()}, flush=True)


def measure_latency(net, base, radar, names_idx, device, n=200):
    """Per-frame time of the fusion forward pass alone (radar feature time is measured separately)."""
    net.eval()
    t = []
    with torch.no_grad():
        for i in range(0, min(n, len(base))):
            x, col = frame_cells(base[i], radar[i], names_idx, device)
            torch.cuda.synchronize()
            t0 = time.time()
            net(x, col, len(base[i]["scores"]))
            torch.cuda.synchronize()
            t.append(time.time() - t0)
    return float(np.median(t) * 1000)


if __name__ == "__main__":
    main()
