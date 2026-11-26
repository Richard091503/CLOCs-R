"""B_sel: the fixed blend B of the paper, with its weights learned on `select` instead of `fit` (PREDICTIONS.md).

B (thesis branch, clocs_r/selector.py, `--blend`) scores every SECOND candidate as sum_k w_k * logit_k over the experts
{L, LC, LRd} (L = logit(s3D)), with one weight vector w = softmax(b) for all candidates, b started at (2, 0, 0). b is
trained on out-of-fold expert logits: for each of 3 folds of whole recordings, LC and LRd are retrained (same seed) on
the other folds and score the held-out fold. At test the experts are the full models of the same seed.

This file is that procedure on master's code, with the training frames, folds and step counts of PREDICTIONS.md:
- folds: splits._fold_of's rule (per condition, recordings from the largest down, each to the fold with the fewest
  frames of that condition so far, ties to the lower index), applied to `select`'s recordings;
- out-of-fold experts: `oof_steps` optimizer steps on the other folds; the blend: `blend_steps` steps over `select`;
- test experts: LC_sel and LRd_sel of the same seed and length (`--experts`).
check_equivalence.py confirms that the training steps here equal the thesis code's, update for update.

    python results/select_trained/blend.py --length primary --seeds 1,2,3 --experts runs/select_trained/primary \
        --out runs/select_trained/primary
"""
import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from clocs_r import bootstrap, cache_cells, fusion, range_weight, splits, train_fusion  # noqa: E402
from clocs_r import tune_fusion as tf  # noqa: E402
from clocs_r.aimotive import Sequence  # noqa: E402

LC, LRD = "R0+T2+K40", "R0+T2+RP"
INIT_BIAS = (2.0, 0.0, 0.0)  # thesis selector.INIT_BIAS: close to LiDAR alone
EXPERTS = ("L", "LC", "LRd")
FOLDS = 3
FIT_STEPS = 53_979  # 3 epochs x 17,993 fit frames: every fit-trained fusion network of the paper
FAR_WEIGHT = 2.0
R = pathlib.Path("runs/lcr")
DATA = {}


class Blend(nn.Module):
    """thesis selector.Blend: w = softmax(b) for every candidate; the output is sum_k w_k * logit_k."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(INIT_BIAS))

    def forward(self, logits):
        return (torch.softmax(self.bias, 0).expand(len(logits), -1) * logits).sum(1)


def fold_of_select():
    """Recording -> fold, by splits._fold_of's rule on select."""
    out = {}
    scenes = splits.scenes("select")
    for odd in splits.ODDS:
        frames = {}
        for s in scenes:
            if s.rstrip(os.sep).split(os.sep)[-2] == odd:
                r = splits.recording(s)
                frames[r] = frames.get(r, 0) + len(Sequence(s).kept)
        load = [0] * FOLDS
        for r, n in sorted(frames.items(), key=lambda kv: (-kv[1], kv[0])):
            k = min(range(FOLDS), key=lambda i: (load[i], i))
            load[k] += n
            out[r] = k
    return out


def fold_indices(items):
    fold = fold_of_select()
    of = np.array([fold[splits.recording(it["seq"])] for it in items])
    return [np.nonzero(of == k)[0] for k in range(FOLDS)]


def steps_for(length, n):
    """Optimizer steps of a network trained on n select frames: primary keeps the fit runs' 15.83 passes
    (FIT_STEPS / 3,409 frames of select, so the full experts take exactly FIT_STEPS), secondary 3 epochs."""
    steps = int(round(FIT_STEPS * n / DATA["n_select"])) if length == "primary" else 3 * n
    return min(steps, DATA["max_steps"]) if DATA.get("max_steps") else steps


def logit_l(items):
    return [np.log(np.clip(it["scores"], 1e-6, 1 - 1e-6) / (1 - np.clip(it["scores"], 1e-6, 1 - 1e-6))).astype(np.float32)
            for it in items]


def expert_inputs(name, d):
    cfg = tf.parse(name)
    return cfg, (d["rp"] if tf.kind(cfg) == "rp" else d["items"]), tf.pairing(cfg, d["top2d"])


@torch.no_grad()
def expert_logits(net, items, pairing, device):
    net.eval()
    out = []
    for i, it in enumerate(items):
        x, col = train_fusion.frame_cells(it, device, pairing[i] if pairing is not None else None)
        out.append(net(x, col, len(it["scores"])).float().cpu().numpy())
    return out


def oof_logits(d, seed, length, device):
    """(n, 3) logits per select frame: L, and LC and LRd from experts trained on the other folds (thesis oof_logits)."""
    out = [None] * len(d["items"])
    ll = logit_l(d["items"])
    steps = {}
    for k, held in enumerate(fold_indices(d["items"])):
        rest = np.setdiff1d(np.arange(len(d["items"])), held)
        tr, te = bootstrap.subset(d, rest), bootstrap.subset(d, held)
        cols = {}
        for e, name in (("LC", LC), ("LRd", LRD)):
            cfg, items, pairing = expert_inputs(name, tr)
            n = steps_for(length, len(items))
            net, _ = train_fusion.fit(items, 3, seed, device, f"oof {name} fold {k} s{seed}", net_kwargs=cfg["net"],
                                      far_weight=FAR_WEIGHT, n_extra=tf.n_extra(cfg), pairing=pairing, steps=n)
            steps[f"{e} fold {k}"] = dict(train_frames=len(items), steps=n)
            cfg, items, pairing = expert_inputs(name, te)
            cols[e] = expert_logits(net, items, pairing, device)
        for j, i in enumerate(held):
            out[i] = np.stack([ll[i], cols["LC"][j], cols["LRd"][j]], 1)
    return out, steps


def train_blend(logits, items, seed, device, steps):
    """thesis selector.train_gate for the blend, with the number of optimizer steps given (3 epochs there)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    gate = Blend().to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.4, div_factor=10,
                                                final_div_factor=1e4)
    gate.train()
    step = 0
    while step < steps:
        for i in rng.permutation(len(items)):
            if step == steps:
                break
            it = items[i]
            w = it["weight"] * range_weight.weight(it["dist"] * cache_cells.MAX_RANGE, far_weight=FAR_WEIGHT).astype(np.float32)
            out = gate(torch.from_numpy(logits[i]).to(device))
            loss = fusion.focal_loss(out, torch.from_numpy(it["target"]).to(device), torch.from_numpy(w.astype(np.float32)).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
    gate.eval()
    return gate


def load_blend(path, device):
    state = torch.load(path, map_location="cpu")
    assert tuple(state["init_bias"]) == INIT_BIAS and state.get("kind", "blend") == "blend", path
    gate = Blend().to(device)
    gate.load_state_dict(state["model"])
    gate.eval()
    return gate


@torch.no_grad()
def blend_scores(gate, d, lc_weights, lrd_weights, seed, device):
    """Sigmoid scores of the blend on the split in d (bootstrap.load_split format), with the given full experts."""
    cols = [logit_l(d["items"])]
    for name, weights in ((LC, lc_weights), (LRD, lrd_weights)):
        cfg, items, pairing = expert_inputs(name, d)
        net = tf.load_net(cfg, name, weights, seed, FAR_WEIGHT, device)
        cols.append(expert_logits(net, items, pairing, device))
    # torch.sigmoid of the blended logit, as the thesis bootstrap.selector_scores and selector.evaluate
    return [torch.sigmoid(gate(torch.from_numpy(np.stack(lg, 1)).to(device))).cpu().numpy() for lg in zip(*cols)]


def run_seed(job):
    seed, length, out = job
    time.sleep(tf.STAGGER_S * (seed - 1) % (3 * tf.STAGGER_S))
    torch.set_num_threads(1)
    device = torch.device("cuda")
    t0 = time.time()
    d = DATA["select"]
    logits, oof_steps = oof_logits(d, seed, length, device)
    n = steps_for(length, len(d["items"]))
    gate = train_blend(logits, d["items"], seed, device, n)
    path = pathlib.Path(out) / f"blend_s{seed}.pth"
    torch.save(dict(model=gate.state_dict(), kind="blend", seed=seed, init_bias=INIT_BIAS, length=length, steps=n,
                    oof=oof_steps, train_split="select"), path)
    w = torch.softmax(gate.bias.detach().cpu(), 0).numpy()
    return dict(seed=seed, weights=dict(zip(EXPERTS, w.round(6).tolist())), bias=gate.bias.detach().cpu().tolist(),
                blend_steps=n, oof=oof_steps, seconds=round(time.time() - t0, 1), path=str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", choices=("primary", "secondary"), required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--processes", type=int, default=3)
    ap.add_argument("--out", required=True, help="folder for blend_s{seed}.pth")
    ap.add_argument("--log", required=True, help="json file the runs are added to")
    ap.add_argument("--max_steps", type=int, default=0, help="tests only: cap every network's steps")
    args = ap.parse_args()
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    DATA["select"] = bootstrap.load_split("select", R / "cells", R / "cand/3d", R / "cand/2d", R / "d2_dets")
    DATA["n_select"] = len(DATA["select"]["items"])
    DATA["max_steps"] = args.max_steps
    seeds = [int(s) for s in args.seeds.split(",")]
    with mp.get_context("fork").Pool(min(args.processes, len(seeds))) as pool:
        runs = pool.map(run_seed, [(s, args.length, args.out) for s in seeds])
    log = pathlib.Path(args.log)
    res = json.load(open(log)) if log.exists() else {}
    key = f"B_sel {args.length}"
    prev = {r["seed"]: r for r in res.get(key, {}).get("runs", [])}
    prev.update({r["seed"]: r for r in runs})
    folds = fold_of_select()
    res[key] = dict(folds={r: k for r, k in sorted(folds.items(), key=lambda kv: (kv[1], kv[0]))},
                    fold_frames=[int(len(i)) for i in fold_indices(DATA["select"]["items"])],
                    runs=[prev[s] for s in sorted(prev)])
    json.dump(res, open(log, "w"), indent=1)
    for r in runs:
        print(key, r["seed"], r["weights"], f"{r['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
