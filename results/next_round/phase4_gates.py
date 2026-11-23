"""Gate values of the Phase 4 gated model by distance bin and condition (results/next_round/PLAN.md, Phase 4).

Candidates: SECOND score >= 0.05 in the scored region. True cars: center within 2 m of a scored car. Writes the means
to results/next_round/phase4/gates_<split>.json and the plot to results/next_round/figures/<out>.

    python results/next_round/phase4_gates.py --configs R2+T2+K40+G,R2+T2+K40+G+C --split select --seeds 1,2,3 --out gates_select.png
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from clocs_r import cache_cells, splits, train_fusion, tune_fusion as tf, visibility

HERE = pathlib.Path(__file__).parent
P1 = "runs/next_round/phase1"
BINS = ((0, 30), (30, 50), (50, 100), (100, 200))
MIN_SCORE, CD = 0.05, 2.0
LABEL_GAP = 0.075  # minimum vertical spacing of the end labels (gate units)
COLORS = dict(highway="#2a78d6", urban="#eb6834", night="#1baf7a", rain="#eda100")  # categorical slots 1-4, fixed order
MARKERS = dict(highway="o", urban="s", night="^", rain="D")


@torch.no_grad()
def collect(name, split, seeds, weights, device):
    cfg = tf.parse(name)
    idx = tf.names_idx(cfg)
    base = tf.DATA[split]
    if cfg["radar"] and "moving_" + split not in tf.DATA:
        tf.DATA["moving_" + split] = visibility.build_moving(split, f"{P1}/cells_in", f"{P1}/cand_in/3d", *tf.GATE, cfg["sweeps"])
    radar = tf.DATA["radar_" + split] if idx else [None] * len(base)
    tf.DATA["ctx_" + split] = tf.gate_context(cfg, split, base, radar)
    extra, pairing = tf.p3_inputs(cfg, split)
    sums = {}  # (condition, bin, true) -> [sum g_cam, sum g_radar, count]
    for seed in seeds:
        state = torch.load(pathlib.Path(weights) / f"{name}_s{seed}.pth", map_location="cpu")
        assert state["settings"] == cfg
        net = tf.make_net(cfg, idx).to(device)
        net.load_state_dict(state["model"])
        net.eval()
        for i, item in enumerate(base):
            x, col = train_fusion.frame_cells(item, radar[i], idx, device, False, extra[i], pairing[i] if pairing else None)
            net(x, col, len(item["scores"]))
            g = net.last_gates.cpu().numpy()
            m = item["keep_eval"] & (item["scores"] >= MIN_SCORE)
            boxes = item["boxes"][m]
            rng = np.hypot(boxes[:, 0], boxes[:, 1])
            gt = item["gt"]
            true = (np.hypot(boxes[:, None, 0] - gt[None, :, 0], boxes[:, None, 1] - gt[None, :, 1]).min(1) <= CD) if len(gt) \
                else np.zeros(len(boxes), bool)
            for lo, hi in BINS:
                inb = (rng >= lo) & (rng < hi)
                for t in (True, False):
                    sel = inb & (true == t)
                    if sel.any():
                        s = sums.setdefault(f"{item['odd']}|{lo}-{hi}|{t}", [0.0, 0.0, 0])
                        s[0] += float(g[m][sel, 0].sum())
                        s[1] += float(g[m][sel, 1].sum())
                        s[2] += int(sel.sum())
    return {k: dict(g_cam=v[0] / v[2], g_radar=v[1] / v[2], n=v[2]) for k, v in sums.items()}


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results)
    fig, axes = plt.subplots(2 * len(names), 2, figsize=(10, 4.2 * len(names)), sharex=True, sharey=True, squeeze=False)
    x = np.arange(len(BINS))
    for r, name in enumerate(names):
        for gi, gate in enumerate(("g_cam", "g_radar")):
            for c, truth in enumerate((True, False)):
                ax = axes[2 * r + gi, c]
                ends = []
                for odd in splits.ODDS:
                    y = [results[name].get(f"{odd}|{lo}-{hi}|{truth}", {}).get(gate, np.nan) for lo, hi in BINS]
                    ax.plot(x, y, color=COLORS[odd], marker=MARKERS[odd], markersize=7, linewidth=2, label=odd)
                    last = max((k for k in range(len(y)) if not np.isnan(y[k])), default=None)
                    if last is not None:
                        ends.append([y[last], x[last], y[last], odd])
                ends.sort()  # spread the end labels at least GAP apart so converging lines stay readable
                for k in range(1, len(ends)):
                    ends[k][0] = max(ends[k][0], ends[k - 1][0] + LABEL_GAP)
                shift = max(0.0, ends[-1][0] - 0.98) if ends else 0.0
                for ly, lx, py, odd in ends:
                    ax.annotate(odd, (lx, py), xytext=(lx + 0.12, ly - shift), textcoords="data", va="center", fontsize=8,
                                color="#3d3d3a", arrowprops=dict(arrowstyle="-", color="#b4b2a9", linewidth=0.6, shrinkA=0, shrinkB=4))
                ax.set_title(f"{name}: {gate} for {'true cars' if truth else 'other candidates'}", fontsize=9, color="#1a1a19", loc="left")
                ax.set_xticks(x, [f"{lo}–{hi} m" for lo, hi in BINS], fontsize=8)
                ax.set_ylim(0, 1)
                ax.set_xlim(-0.3, len(BINS) - 0.25)
                ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
                ax.spines[["top", "right"]].set_visible(False)
                ax.spines[["left", "bottom"]].set_color("#b4b2a9")
                ax.tick_params(colors="#5f5e5a", labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Mean gate value by distance and driving condition (1 = the term is used fully)", fontsize=10, x=0.01, ha="left")
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0.01, 0.985 - 0.01 / len(names)))
    fig.tight_layout(rect=(0, 0, 1, 0.985 - 0.03 / len(names)))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--weights", default="runs/next_round/phase4/fusion")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    s = args.split
    tf.DATA[s], _ = cache_cells.load(s, f"{P1}/cells_in", *tf.GATE, cache_cells.SWEEPS, use_radar=False)
    tf.DATA["vis_" + s] = visibility.load(s, f"{P1}/cells_in", tf.DATA[s])
    tf.DATA["radar_" + s] = cache_cells.load_radar(s, f"{P1}/cells_in", *tf.GATE, cache_cells.SWEEPS)
    seeds = [int(x) for x in args.seeds.split(",")]
    results = {name: collect(name, s, seeds, args.weights, device) for name in args.configs.split(",")}
    out = HERE / "phase4" / f"gates_{s}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.load(open(out)) if out.exists() else {}
    json.dump({**old, **results}, open(out, "w"), indent=1)
    plot(results, HERE / "figures" / args.out)
    print("wrote", out, HERE / "figures" / args.out)


if __name__ == "__main__":
    main()
