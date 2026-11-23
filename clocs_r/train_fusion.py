"""Train and evaluate one CLOCs fusion network from cached cells (used by tune_fusion, which handles the configurations).

A frame's input is one row per (SECOND candidate, paired detection) cell:
    [IoU, 2D score, s3D, distance]                                  camera CLOCs (LC) and radar CLOCs (LRd)
    [IoU, 2D score, s3D, distance, affinity, radar score]           LCR, one network over all three detectors
The network scores every cell and keeps the maximum per candidate (fusion.FusionNet).
"""
import time

import numpy as np
import torch

from clocs_r import aimotive, cache_cells, fusion, metrics, range_weight, splits

FAR = (50, 200)  # the 50-100 m and 100-200 m bins scored together


def frame_cells(item, device, pairing=None):
    """The (P, 4 + e) input of one frame and the candidate index of each row.

    pairing: optional (min_iou, s2d_floor) for fusion.filter_pairs (the K40 camera floor of LC).
    """
    column, pairs = item["column"], item["pairs"]
    if pairing is not None:
        column, pairs = fusion.filter_pairs(column, pairs, len(item["scores"]), *pairing)
    pairs = pairs.astype(np.float32)  # [IoU, 2D score], or with token RC [IoU, 2D score, affinity, radar score]
    cells = np.concatenate([pairs[:, :2], item["scores"][column, None], item["dist"][column, None], pairs[:, 2:]], 1)
    x = torch.from_numpy(cells).to(device)
    col = torch.from_numpy(np.asarray(column, dtype=np.int64)).to(device)
    return x, col


@torch.no_grad()
def predict(net, base, device, pairing=None, nms_thresh=0.1):
    """Yields (item, kept boxes, scores) after fused scoring (SECOND's own scores if net is None) and NMS."""
    was_training = net.training if net is not None else False
    if net is not None:
        net.eval()
    for i, item in enumerate(base):
        if net is None:
            scores = item["scores"]
        else:
            x, col = frame_cells(item, device, pairing[i] if pairing is not None else None)
            scores = torch.sigmoid(net(x, col, len(item["scores"]))).cpu().numpy()
        mask = item["keep_eval"]
        boxes, s = item["boxes"][mask], scores[mask]
        keep = metrics.nms_bev(boxes, s, nms_thresh)
        yield item, boxes[keep], s[keep]
    if was_training:
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


def evaluate(net, base, device, pairing=None):
    """Metrics by distance bin (with 50-200 m as one more bin), overall and for each group of driving conditions."""
    records = [(it["gt"], b, s, it["odd"], it["seq"]) for it, b, s in predict(net, base, device, pairing)]
    return {name: metrics.evaluate([r[:3] for r in rows], bins=aimotive.DIST_BINS + [FAR])
            for name, rows in condition_groups(records).items()}


def fit(train, epochs, seed, device, label="", net_kwargs=None, far_weight=1.0, n_extra=0, pairing=None,
        far_cam_dropout=None):
    """Train a fusion network on cached frames. Returns the network and its loss log.

    far_weight: weight of candidates beyond 60 m in the loss, ramping up from 40 m (range_weight.py); 1 leaves it unweighted.
    n_extra: input channels after the four CLOCs ones (2 for token RC: affinity and radar score).
    pairing: per-frame (min_iou, s2d_floor) for frame_cells.
    far_cam_dropout: (normalized distance, p) for token DF (LCR-D): at each step, each candidate beyond the distance loses
    its camera cells with probability p (fusion.drop_far_camera), from a random stream separate from the frame order.
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    drop_rng = np.random.RandomState(seed + 10_000)
    net = fusion.FusionNet(n_extra, **(net_kwargs or {})).to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    steps = epochs * len(train)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=3e-3, total_steps=steps, pct_start=0.4, div_factor=10,
                                                final_div_factor=1e4, cycle_momentum=True, base_momentum=0.85, max_momentum=0.95)
    log, t0, step, running = [], time.time(), 0, 0.0
    net.train()
    for epoch in range(epochs):
        for i in rng.permutation(len(train)):
            item = train[i]
            if far_cam_dropout is not None:
                column, pairs = fusion.drop_far_camera(item["column"], item["pairs"], item["dist"] > far_cam_dropout[0],
                                                       far_cam_dropout[1], drop_rng)
                item = dict(item, column=column, pairs=pairs)
            x, col = frame_cells(item, device, pairing[i] if pairing is not None else None)
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
