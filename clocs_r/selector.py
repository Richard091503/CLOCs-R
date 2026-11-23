"""Situational selector over CLOCs experts (results/next_round/PLAN.md, Phase 7).

For each SECOND candidate, w = softmax(g(context)) weights the experts' logits {L, LC, LR*} (L = logit(s3D), LC the
camera CLOCs model R0+T2+K40, LR* the chosen radar CLOCs model). Output = sum_k w_k * logit_k. g is a small MLP whose last
layer starts at zero weights and bias INIT_BIAS, i.e. close to LiDAR alone. g is trained on fit with out-of-fold expert
logits: for each of fit's scene folds, each trained expert is retrained on the other folds (same seed) and scores the
held-out fold. At test the experts are the full-fit models of the same seed.

    python -m clocs_r.selector --radar_expert R2+T2+NC --radar_weights runs/next_round/phase7/fusion --seeds 1,2,3,4,5,6 \
        --splits select --out results/next_round/phase7/selector_select.json
"""
import argparse
import json
import multiprocessing as mp
import pathlib

import numpy as np
import torch
from torch import nn

from clocs_r import aimotive, cache_cells, fusion, metrics, radar_pairs, range_weight, splits, train_fusion, visibility
from clocs_r import tune_fusion as tf

CAMERA_EXPERT, CAMERA_WEIGHTS = "R0+T2+K40", "runs/next_round/phase3/fusion"
EXPERTS = ("L", "LC", "LR")
INIT_BIAS = (2.0, 0.0, 0.0)
N_CONTEXT = 10  # distance, 3 logits, camera evidence, radar returns, moving returns, LiDAR points, projected height, s3D
P1 = "runs/next_round/phase1"
FOLDS = 3


class Gate(nn.Module):
    def __init__(self, n_context=N_CONTEXT, hidden=16, n_experts=len(EXPERTS)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_context, hidden), nn.ReLU(), nn.Linear(hidden, n_experts))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor(INIT_BIAS[:n_experts]))
        self.last_weights = None

    def forward(self, context, logits):
        w = torch.softmax(self.net(context), 1)
        self.last_weights = w.detach()
        return (w * logits).sum(1)


class Blend(nn.Module):
    """Post-hoc ablation of the Gate (PLAN.md, Phase 7 blend addendum): the same output, but w = softmax(b) for every
    candidate, with b trained and started at INIT_BIAS. The context is accepted and ignored."""

    def __init__(self, n_experts=len(EXPERTS)):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(INIT_BIAS[:n_experts]))
        self.last_weights = None

    def forward(self, context, logits):
        w = torch.softmax(self.bias, 0).expand(len(logits), -1)
        self.last_weights = w.detach()
        return (w * logits).sum(1)


KINDS = {"selector": Gate, "blend": Blend}
PREFIX = {"selector": "S", "blend": "B"}


def gate_path(out_dir, radar_expert, seed, kind="selector"):
    return pathlib.Path(out_dir) / f"{kind}_{radar_expert}_s{seed}.pth"


def camera_evidence(item, top2d):
    """Per candidate: the highest IoU x 2D score over its K40 camera pairs, or MISSING without one."""
    column, pairs = fusion.filter_pairs(item["column"], item["pairs"], len(item["scores"]), 0.0, top2d)
    ev = np.full(len(item["scores"]), fusion.MISSING, np.float32)
    p = pairs.astype(np.float32)
    real = p[:, 0] != fusion.MISSING
    np.maximum.at(ev, column[real], p[real, 0] * p[real, 1])
    return ev


def context(item, logits, vis, radar, moving, top2d):
    """(n, N_CONTEXT) float32 gate input of one frame; logits (n, 3) in EXPERTS order."""
    s3d = item["scores"].astype(np.float32)
    lg = np.clip(logits, -12, 12) / 6.0
    return np.column_stack([item["dist"], lg, camera_evidence(item, top2d), np.asarray(radar, np.float32)[:, 0],
                            np.asarray(moving, np.float32), np.asarray(vis, np.float32)[:, 1], np.asarray(vis, np.float32)[:, 0],
                            s3d]).astype(np.float32)


@torch.no_grad()
def expert_logits(net, cfg, items, radar, extra, pairing, device):
    idx = tf.names_idx(cfg)
    nocam = bool(cfg.get("nocam"))
    out = []
    for i, item in enumerate(items):
        x, col = train_fusion.frame_cells(item, radar[i] if idx else None, idx, device, nocam,
                                          extra[i] if extra is not None else None, pairing[i] if pairing is not None else None)
        out.append(net(x, col, len(item["scores"])).float().cpu().numpy())
    return out


def logit_l(items):
    return [np.log(np.clip(it["scores"], 1e-6, 1 - 1e-6) / (1 - np.clip(it["scores"], 1e-6, 1 - 1e-6))).astype(np.float32) for it in items]


def inputs(cfg, data, rp_items=None):
    """Items, radar, extra and pairing an expert sees, from the split's data dict."""
    items = rp_items if cfg.get("rp") else data["items"]
    pairing = [(0.0, t) for t in data["vis"]["top2d"]] if cfg.get("p3") else None
    return items, data["radar"], None, pairing


def fit_expert(cfg, items, radar, pairing, seed, device, far_weight=2.0):
    idx = tf.names_idx(cfg)
    net, _ = train_fusion.fit(items, radar, idx, 3, seed, device, f"oof {seed}", net_kwargs=cfg["net"], far_weight=far_weight,
                              pairing=pairing, cam_dropout=1.0 if cfg.get("nocam") else 0.0, model=lambda: tf.make_net(cfg, idx))
    net.eval()
    return net


def load_expert(name, weights, seed, device):
    cfg = tf.parse(name)
    state = torch.load(pathlib.Path(weights) / f"{name}_s{seed}.pth", map_location="cpu")
    assert state["settings"] == cfg and state.get("far_weight") == 2.0, f"{name} s{seed}"
    net = tf.make_net(cfg, tf.names_idx(cfg)).to(device)
    net.load_state_dict(state["model"])
    net.eval()
    return cfg, net


def frame_contexts(data, logits):
    return [context(it, lg, v, r, m, t) for it, lg, v, r, m, t in
            zip(data["items"], logits, data["vis"]["features"], data["radar"], data["moving"], data["vis"]["top2d"])]


def train_gate(ctx, logits, items, seed, device, epochs=3, far_weight=2.0, kind="selector"):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    gate = KINDS[kind]().to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * len(items), pct_start=0.4, div_factor=10,
                                                final_div_factor=1e4)
    gate.train()
    for _ in range(epochs):
        for i in rng.permutation(len(items)):
            it = items[i]
            w = it["weight"] * range_weight.weight(it["dist"] * cache_cells.MAX_RANGE, far_weight=far_weight).astype(np.float32)
            out = gate(torch.from_numpy(ctx[i]).to(device), torch.from_numpy(logits[i]).to(device))
            loss = fusion.focal_loss(out, torch.from_numpy(it["target"]).to(device), torch.from_numpy(w.astype(np.float32)).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    gate.eval()
    return gate


@torch.no_grad()
def evaluate(gate, ctx, logits, items, device):
    scores, weights = [], []
    for c, lg in zip(ctx, logits):
        scores.append(torch.sigmoid(gate(torch.from_numpy(c).to(device), torch.from_numpy(lg).to(device))).cpu().numpy())
        weights.append(gate.last_weights.cpu().numpy())
    records = []
    for it, s in zip(items, scores):
        m = it["keep_eval"]
        b, ss = it["boxes"][m], s[m]
        keep = metrics.nms_bev(b, ss, 0.1)
        records.append((it["gt"], b[keep], ss[keep], it["odd"], it["seq"]))
    res = {name: metrics.evaluate([r[:3] for r in rows], bins=aimotive.DIST_BINS + [train_fusion.FAR])
           for name, rows in train_fusion.condition_groups(records).items()}
    return res, weights


DATA = {}


def load_split(split, cells, cand3d, dets_rp=None):
    items, _ = cache_cells.load(split, cells, *tf.GATE, cache_cells.SWEEPS, use_radar=False)
    d = dict(items=items, radar=cache_cells.load_radar(split, cells, *tf.GATE, cache_cells.SWEEPS),
             vis=visibility.load(split, cells, items), moving=visibility.build_moving(split, cells, cand3d, *tf.GATE))
    rp = radar_pairs.build(split, cells, dets_rp, items) if dets_rp else None
    d["rp_items"] = radar_pairs.replace_pairs(items, rp) if dets_rp else None
    d["rc_items"] = radar_pairs.add_pairs(items, rp, d["vis"]["top2d"]) if dets_rp else None  # token RC (LCR)
    return d


def subset(d, idx):
    s = dict(items=[d["items"][i] for i in idx], radar=[d["radar"][i] for i in idx], moving=[d["moving"][i] for i in idx],
             vis=dict(features=[d["vis"]["features"][i] for i in idx], top2d=d["vis"]["top2d"][idx]))
    for k in ("rp_items", "rc_items"):
        s[k] = [d[k][i] for i in idx] if d.get(k) is not None else None
    return s


def fold_indices(items):
    key = {(it["seq"], it["frame"]): i for i, it in enumerate(items)}
    folds = [np.array([key[(s.name, f)] for s, f in splits.frames(f"fold{k}")]) for k in range(FOLDS)]
    assert sorted(np.concatenate(folds).tolist()) == list(range(len(items))), "the folds must partition fit"
    return folds


def oof_logits(fit, radar_expert, seed, device):
    """(n, 3) logits per fit frame: L, and LC and LR from experts trained on the other folds."""
    out = [None] * len(fit["items"])
    l_logits = logit_l(fit["items"])
    cfgs = {"LC": tf.parse(CAMERA_EXPERT), "LR": tf.parse(radar_expert)}
    for k, held in enumerate(fold_indices(fit["items"])):
        rest = np.setdiff1d(np.arange(len(fit["items"])), held)
        tr, te = subset(fit, rest), subset(fit, held)
        cols = {}
        for e, cfg in cfgs.items():
            items, radar, _, pairing = inputs(cfg, tr, tr["rp_items"])
            net = fit_expert(cfg, items, radar, pairing, seed, device)
            items, radar, extra, pairing = inputs(cfg, te, te["rp_items"])
            cols[e] = expert_logits(net, cfg, items, radar, extra, pairing, device)
        for j, i in enumerate(held):
            out[i] = np.stack([l_logits[i], cols["LC"][j], cols["LR"][j]], 1)
    return out


def test_logits(d, radar_expert, radar_weights, seed, device):
    l_logits = logit_l(d["items"])
    cols = {}
    for e, (name, w) in {"LC": (CAMERA_EXPERT, CAMERA_WEIGHTS), "LR": (radar_expert, radar_weights)}.items():
        cfg, net = load_expert(name, w, seed, device)
        items, radar, extra, pairing = inputs(cfg, d, d["rp_items"])
        cols[e] = expert_logits(net, cfg, items, radar, extra, pairing, device)
    return [np.stack([l, c, r], 1) for l, c, r in zip(l_logits, cols["LC"], cols["LR"])]


def run_seed(job):
    seed, radar_expert, radar_weights, eval_splits, out_dir, from_weights, kind = job
    import time
    time.sleep(tf.STAGGER_S * (seed - 1) % (3 * tf.STAGGER_S))  # one CUDA start-up at a time (tune_fusion.STAGGER_S)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    path = gate_path(out_dir, radar_expert, seed, kind)
    if from_weights:  # the frozen gate, e.g. for the single val scoring at the end of Phase 7
        gate = load_gate(out_dir, radar_expert, seed, device, kind)
    else:
        fit_logits = oof_logits(DATA["fit"], radar_expert, seed, device)
        gate = train_gate(frame_contexts(DATA["fit"], fit_logits), fit_logits, DATA["fit"]["items"], seed, device, kind=kind)
        torch.save(dict(model=gate.state_dict(), radar_expert=radar_expert, seed=seed, init_bias=INIT_BIAS, kind=kind), path)
    result = dict(seed=seed)
    for split in eval_splits:
        lg = test_logits(DATA[split], radar_expert, radar_weights, seed, device)
        res, weights = evaluate(gate, frame_contexts(DATA[split], lg), lg, DATA[split]["items"], device)
        result[split] = res
        result[split + "_weights"] = gate_summary(DATA[split]["items"], weights)
    return result


def load_gate(out_dir, radar_expert, seed, device, kind="selector"):
    state = torch.load(gate_path(out_dir, radar_expert, seed, kind), map_location="cpu")
    assert state["radar_expert"] == radar_expert and tuple(state["init_bias"]) == INIT_BIAS
    assert state.get("kind", "selector") == kind, f"{kind} s{seed}"
    gate = KINDS[kind]().to(device)
    gate.load_state_dict(state["model"])
    gate.eval()
    return gate


def gate_summary(items, weights, bins=((0, 30), (30, 50), (50, 100), (100, 200))):
    """Mean expert weights of candidates with SECOND score >= 0.05 in the scored region, by condition and distance bin."""
    acc = {}
    for it, w in zip(items, weights):
        m = it["keep_eval"] & (it["scores"] >= 0.05)
        r = np.hypot(it["boxes"][m, 0], it["boxes"][m, 1])
        for lo, hi in bins:
            sel = (r >= lo) & (r < hi)
            if sel.any():
                a = acc.setdefault(f"{it['odd']}|{lo}-{hi}", [np.zeros(len(EXPERTS)), 0])
                a[0] += w[m][sel].sum(0)
                a[1] += int(sel.sum())
    return {k: dict(zip(EXPERTS, (v[0] / v[1]).round(4).tolist()), n=v[1]) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radar_expert", required=True)
    ap.add_argument("--radar_weights", required=True)
    ap.add_argument("--radar_dets", default="", help="radar-detector boxes, needed when the radar expert is an RP model")
    ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--splits", default="select")
    ap.add_argument("--processes", type=int, default=3)
    ap.add_argument("--cells", default=f"{P1}/cells_in")
    ap.add_argument("--cand3d", default=f"{P1}/cand_in/3d")
    ap.add_argument("--out", required=True)
    ap.add_argument("--from_weights", action="store_true", help="score the saved gates instead of training them")
    ap.add_argument("--blend", action="store_true", help="the post-hoc fixed-blend ablation B (PLAN.md, Phase 7 addendum)")
    args = ap.parse_args()
    kind = "blend" if args.blend else "selector"
    rp = args.radar_dets if tf.parse(args.radar_expert).get("rp") else None
    out_dir = pathlib.Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ([] if args.from_weights else ["fit"]) + args.splits.split(","):
        DATA[split] = load_split(split, args.cells, args.cand3d, rp)
    seeds = [int(s) for s in args.seeds.split(",")]
    with mp.get_context("fork").Pool(min(args.processes, len(seeds))) as pool:
        runs = pool.map(run_seed, [(s, args.radar_expert, args.radar_weights, args.splits.split(","), out_dir, args.from_weights, kind)
                                   for s in seeds])
    out = pathlib.Path(args.out)
    results = json.load(open(out)) if out.exists() else {}
    results[f"{PREFIX[kind]}({args.radar_expert})"] = dict(kind=kind, from_weights=args.from_weights, radar_expert=args.radar_expert, camera_expert=CAMERA_EXPERT, init_bias=list(INIT_BIAS), runs=runs)
    json.dump(results, open(out, "w"), indent=1)
    for split in args.splits.split(","):
        print(split, np.mean([r[split]["overall"]["50-200"]["APCD"] for r in runs]), flush=True)


if __name__ == "__main__":
    main()
