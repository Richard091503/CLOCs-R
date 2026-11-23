"""Oracle ceiling of situation-dependent expert blends on select (results/next_round/PLAN.md, Phase 7 ceiling addendum).

For each seed, the full-fit experts' logits z = (L, LC, LR*) on select are blended as sigmoid(sum_k w_k z_k) for every w
on a simplex grid. NMS and APCD are exactly metrics.evaluate's (greedy BEV NMS at 0.1 over the top 1,000, at most 100
kept; greedy centre-distance matching in score order within each bin), compiled with numba over precomputed overlaps.
Oracles, all tuned on select itself (upper bounds, never model results):
  global       the best single w for each bin;
  distance     the best w per distance bin, each candidate scored with its own bin's w, evaluated once;
  dist x cond  the same per condition label (annotations, so an upper bound only).

    python -m clocs_r.selector_ceiling --radar_dets runs/next_round/phase7/d2_dets --out results/next_round/phase7/ceiling.json
"""
import argparse
import json
import pathlib

import numba
import numpy as np
import torch

from clocs_r import metrics, selector

DIST = [(0, 30), (30, 50), (50, 100), (100, 200)]
EVAL = DIST + [(50, 200)]
NAMES = [f"{lo}-{hi}" for lo, hi in EVAL]
P1 = "runs/next_round/phase1"


def simplex(step=0.05):
    """Every (w_L, w_LC, w_LR) >= 0 summing to 1, in steps of step."""
    n = int(round(1 / step))
    return np.array([(a, b, n - a - b) for a in range(n + 1) for b in range(n + 1 - a)], np.float64) / n


def prepare(items):
    """Flattened per-frame arrays: candidates in the scored region, their BEV overlaps above the NMS threshold, and GT."""
    fstart, cx, cy, nbr_ptr, nbr_idx, gstart, gx, gy, keep_idx = [0], [], [], [0], [], [0], [], [], []
    for it in items:
        m = np.flatnonzero(it["keep_eval"])
        b = it["boxes"][m]
        keep_idx.append(m)
        cx.append(b[:, 0]), cy.append(b[:, 1])
        if len(b):
            iou = metrics.iou_bev(b, b)
            for i in range(len(b)):
                j = np.flatnonzero(iou[i] > 0.1)
                nbr_idx.append(j[j != i])
                nbr_ptr.append(nbr_ptr[-1] + len(j[j != i]))
        fstart.append(fstart[-1] + len(b))
        gx.append(it["gt"][:, 0]), gy.append(it["gt"][:, 1])
        gstart.append(gstart[-1] + len(it["gt"]))
    cat = lambda xs, t: np.concatenate(xs).astype(t) if xs else np.zeros(0, t)
    return dict(fstart=np.array(fstart, np.int64), cx=cat(cx, np.float64), cy=cat(cy, np.float64),
                nbr_ptr=np.array(nbr_ptr, np.int64), nbr_idx=cat(nbr_idx, np.int64), gstart=np.array(gstart, np.int64),
                gx=cat(gx, np.float64), gy=cat(gy, np.float64), keep_idx=keep_idx)


@numba.njit(cache=True)
def _score(scores, fstart, cx, cy, nbr_ptr, nbr_idx, gstart, gx, gy, lo, hi, thr, pre_max, post_max):
    F = len(fstart) - 1
    nb, nt = len(lo), len(thr)
    o_s, o_f = np.empty(F * post_max), np.empty(F * post_max, np.int64)
    o_r = np.empty(F * post_max)
    o_tp = np.zeros((F * post_max, nb, nt), np.bool_)
    k = 0
    for f in range(F):
        a, b = fstart[f], fstart[f + 1]
        n = b - a
        if n == 0:
            continue
        order = np.argsort(-scores[a:b])[:pre_max]
        alive = np.zeros(n, np.bool_)
        for p in range(len(order)):
            alive[order[p]] = True
        kept = np.empty(post_max, np.int64)
        nk = 0
        for p in range(len(order)):
            i = order[p]
            if alive[i]:
                kept[nk] = i
                nk += 1
                if nk == post_max:
                    break
                for q in range(nbr_ptr[a + i], nbr_ptr[a + i + 1]):
                    alive[nbr_idx[q]] = False
        g0, g1 = gstart[f], gstart[f + 1]
        grng = np.sqrt(gx[g0:g1] ** 2 + gy[g0:g1] ** 2)
        used = np.zeros((nb, nt, g1 - g0), np.bool_)
        for p in range(nk):
            i = kept[p]
            x, y = cx[a + i], cy[a + i]
            r = np.sqrt(x * x + y * y)
            o_s[k], o_f[k], o_r[k] = scores[a + i], f, r
            for bi in range(nb):
                if r < lo[bi] or r >= hi[bi]:
                    continue
                for ti in range(nt):
                    best, bd = -1, 1e18
                    for g in range(g1 - g0):
                        if used[bi, ti, g] or grng[g] < lo[bi] or grng[g] >= hi[bi]:
                            continue
                        dd = np.sqrt((gx[g0 + g] - x) ** 2 + (gy[g0 + g] - y) ** 2)
                        if dd <= thr[ti] and dd < bd:
                            best, bd = g, dd
                    if best >= 0:
                        used[bi, ti, best] = True
                        o_tp[k, bi, ti] = True
            k += 1
    return o_s[:k], o_f[:k], o_r[:k], o_tp[:k]


class Scorer:
    def __init__(self, items, conditions=None):
        self.p = prepare(items)
        self.lo = np.array([b[0] for b in EVAL], np.float64)
        self.hi = np.array([b[1] for b in EVAL], np.float64)
        self.thr = np.array(metrics.CD_THRESHOLDS, np.float64)
        grng = np.hypot(self.p["gx"], self.p["gy"])
        gframe = np.repeat(np.arange(len(items)), np.diff(self.p["gstart"]))
        self.gt_in = np.stack([(grng >= lo) & (grng < hi) for lo, hi in EVAL], 1)  # (G, bins)
        self.gframe = gframe
        self.frame_cond = np.array(conditions if conditions is not None else ["all"] * len(items))
        self.cand_bin = np.digitize(np.hypot(self.p["cx"], self.p["cy"]), [b[1] for b in DIST[:-1]])  # 0..3 by distance

    def flat(self, per_frame):
        """Per-frame arrays over all candidates -> the flattened scored-region candidates."""
        return np.concatenate([np.asarray(z)[m] for z, m in zip(per_frame, self.p["keep_idx"])])

    def run(self, scores):
        p = self.p
        return _score(np.asarray(scores, np.float64), p["fstart"], p["cx"], p["cy"], p["nbr_ptr"], p["nbr_idx"], p["gstart"],
                      p["gx"], p["gy"], self.lo, self.hi, self.thr, 1000, 100)

    def apcd(self, out, frames=None):
        """{bin: APCD} over all frames, or a boolean frame mask."""
        s, f, _, tp = out
        res = {}
        dm = np.ones(len(s), bool) if frames is None else frames[f]
        gm = np.ones(len(self.gframe), bool) if frames is None else frames[self.gframe]
        for bi, name in enumerate(NAMES):
            sel = dm & tp_in_bin(out, self.lo[bi], self.hi[bi])
            ss = s[sel]
            order = np.argsort(-ss)
            n_gt = int((self.gt_in[:, bi] & gm).sum())
            res[name] = float(np.mean([metrics._ap(tp[sel, bi, t][order], n_gt) for t in range(len(self.thr))]))
        return res


def tp_in_bin(out, lo, hi):
    r = out[2]
    return (r >= lo) & (r < hi)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def seed_ceiling(sc, z, b_weights, s_scores, grid):
    """z (N, 3) flattened logits; b_weights (3,); s_scores (N,) the selector's scores. All oracles for one seed."""
    conds = sorted(set(sc.frame_cond))
    masks = {c: sc.frame_cond == c for c in conds}
    ref = dict(B=sc.apcd(sc.run(sigmoid(z @ b_weights))), S=sc.apcd(sc.run(s_scores)))
    table = np.zeros((len(grid), 1 + len(conds), len(NAMES)))
    for gi, w in enumerate(grid):
        out = sc.run(sigmoid(z @ w))
        table[gi, 0] = [sc.apcd(out)[n] for n in NAMES]
        for ci, c in enumerate(conds):
            r = sc.apcd(out, masks[c])
            table[gi, 1 + ci] = [r[n] for n in NAMES]
    glob = {n: float(np.nanmax(table[:, 0, i])) for i, n in enumerate(NAMES)}
    glob_w = {n: grid[int(np.nanargmax(table[:, 0, i]))].tolist() for i, n in enumerate(NAMES)}
    dw = np.stack([grid[int(np.nanargmax(table[:, 0, i]))] for i in range(len(DIST))])  # (4, 3)
    dist = sc.apcd(sc.run(sigmoid((z * dw[sc.cand_bin]).sum(1))))
    cand_frame = np.repeat(np.arange(len(sc.frame_cond)), np.diff(sc.p["fstart"]))
    cw = np.zeros_like(z)
    cond_w = {}
    for ci, c in enumerate(conds):
        wc = np.stack([grid[int(np.nanargmax(np.nan_to_num(table[:, 1 + ci, i], nan=-1)))] for i in range(len(DIST))])
        cond_w[c] = wc.tolist()
        m = sc.frame_cond[cand_frame] == c
        cw[m] = wc[sc.cand_bin[m]]
    dc = sc.apcd(sc.run(sigmoid((z * cw).sum(1))))
    return dict(B=ref["B"], S=ref["S"], global_oracle=glob, global_oracle_w=glob_w, distance_oracle=dist,
                distance_w=dict(zip(NAMES[:4], dw.tolist())), distance_condition_oracle=dc, distance_condition_w=cond_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radar_expert", default="R0+T2+RP")
    ap.add_argument("--radar_weights", default="runs/next_round/phase7/fusion_rp")
    ap.add_argument("--radar_dets", required=True)
    ap.add_argument("--gates", default="results/next_round/phase7")
    ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    d = selector.load_split("select", f"{P1}/cells_in", f"{P1}/cand_in/3d", args.radar_dets)
    if args.limit:
        d = selector.subset(d, np.arange(args.limit))
    sc = Scorer(d["items"], [it["odd"] for it in d["items"]])
    grid = simplex(args.step)
    runs = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        lg = selector.test_logits(d, args.radar_expert, args.radar_weights, seed, device)
        blend = selector.load_gate(args.gates, args.radar_expert, seed, "cpu", "blend")
        bw = torch.softmax(blend.bias, 0).detach().numpy().astype(np.float64)
        gate = selector.load_gate(args.gates, args.radar_expert, seed, device, "selector")
        ctx = selector.frame_contexts(d, lg)
        with torch.no_grad():
            s_sc = [torch.sigmoid(gate(torch.from_numpy(c).to(device), torch.from_numpy(z).to(device))).cpu().numpy()
                    for c, z in zip(ctx, lg)]
        r = seed_ceiling(sc, sc.flat(lg).astype(np.float64), bw, sc.flat(s_sc).astype(np.float64), grid)
        r["seed"], r["B_weights"] = seed, bw.tolist()
        runs.append(r)
        print(f"seed {seed}: " + " | ".join(f"{k} {r[k]['100-200']:.4f}/{r[k]['50-200']:.4f}" for k in
                                            ("B", "S", "global_oracle", "distance_oracle", "distance_condition_oracle")), flush=True)
    mean = {k: {n: float(np.mean([r[k][n] for r in runs])) for n in NAMES}
            for k in ("B", "S", "global_oracle", "distance_oracle", "distance_condition_oracle")}
    gain = mean["distance_oracle"]["100-200"] - mean["B"]["100-200"]
    out = dict(split="select", frames=len(d["items"]), grid_step=args.step, grid_points=len(grid), runs=runs, mean=mean,
               distance_oracle_minus_B=gain, distance_minus_global={n: mean["distance_oracle"][n] - mean["global_oracle"][n] for n in NAMES},
               rule="further selector variants only if distance_oracle - B >= 0.005 at 100-200 m (PLAN.md)",
               proceed=bool(gain >= 0.005))
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    for k, v in mean.items():
        print(k, {n: round(x, 4) for n, x in v.items()}, flush=True)
    print(f"distance oracle - B at 100-200: {gain:+.4f}; proceed: {gain >= 0.005}", flush=True)


if __name__ == "__main__":
    main()
