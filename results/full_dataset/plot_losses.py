"""Training loss curves for SECOND, Cascade R-CNN and the fusion network, from the logs kept in this directory.

    python results/full_dataset/plot_losses.py [--fusion R2+T2,R0+T2]

Inputs (all in git): logs/second_train.txt (SECOND's log), logs/cascade_train.json (MMDetection's log) and the fusion runs'
loss logs inside the tune_fusion result files (ablation/*.json, then the first round's selection.json and rerun files).
By default the fusion plot shows the final CLOCs-R and CLOCs of ablation/decisions.json.
Writes figures/loss_second.png, figures/loss_cascade.png and figures/loss_fusion.png.

All curves are training losses. SECOND's and the fusion network's are range weighted (x2 beyond 60 m), so they are not
comparable with unweighted runs of the same models.
"""
import argparse
import json
import pathlib
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
# Reference palette (dataviz skill), light mode
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
FUSION_SOURCES = ("ablation/extra_seeds.json", "ablation/changes.json", "ablation/feature_sets.json", "ablation/twin.json",
                  "selection.json", "rerun_r3.json", "rerun_r2.json")


def style(ax, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def smooth(y, n):
    """Centered moving average over n points (shorter at the ends)."""
    y = np.asarray(y, float)
    if len(y) < 3 or n < 2:
        return y
    k = np.ones(n) / n
    num = np.convolve(y, k, mode="same")
    den = np.convolve(np.ones_like(y), k, mode="same")
    return num / den


def finish(fig, axes, title, path):
    axes[-1].set_xlabel("Epoch", color=INK_2, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def second():
    pat = re.compile(r"epoch (\d+)/(\d+) iter (\d+)/(\d+) loss ([\d.]+) cls ([\d.]+) loc ([\d.]+) lr ([\d.e-]+)")
    rows = [m.groups() for m in map(pat.search, open(HERE / "logs" / "second_train.txt")) if m]
    it = np.array([int(r[2]) for r in rows])
    total_iters, epochs = int(rows[0][3]), int(rows[0][1])
    x = it / (total_iters / epochs)
    loss, cls, loc, lr = (np.array([float(r[i]) for r in rows]) for i in (4, 5, 6, 7))
    n = max(3, len(x) // 200)  # about 0.2 epoch
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), facecolor=SURFACE, sharex=True, gridspec_kw=dict(height_ratios=(3, 3, 1.4)))
    axes[0].plot(x, loss, color=SERIES[0], alpha=0.18, linewidth=0.6, zorder=2)
    axes[0].plot(x, smooth(loss, n), color=SERIES[0], linewidth=2, zorder=3, label="total incl. direction (smoothed)")
    style(axes[0], "Total loss")
    axes[0].set_ylim(0, np.percentile(loss, 99.5))
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2)
    # The log has the total as a 50-iteration mean but these two as single-iteration values (no direction term), so the
    # direction loss cannot be recovered by subtraction
    for i, (name, y) in enumerate((("classification (focal)", cls), ("box regression", loc))):
        axes[1].plot(x, smooth(y, 5 * n), color=SERIES[i], linewidth=2, label=name, zorder=3)  # about one epoch
    style(axes[1], "Loss terms (smoothed over 1 epoch)")
    axes[1].set_ylim(0, None)
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=2)
    axes[2].plot(x, lr, color=INK_2, linewidth=1.5)
    style(axes[2], "Learning rate")
    finish(fig, axes, f"SECOND training loss, {epochs} epochs on aiMotive train (one-cycle schedule, range-weighted)",
           HERE / "figures" / "loss_second.png")


def cascade():
    rows = [json.loads(line) for line in open(HERE / "logs" / "cascade_train.json")]
    rows = [r for r in rows if r.get("mode") == "train"]
    per_epoch = max(r["iter"] for r in rows)
    per_epoch = int(np.ceil(per_epoch / 100) * 100)  # last logged iteration rounds up to the epoch length
    x = np.array([r["epoch"] - 1 + r["iter"] / per_epoch for r in rows])
    loss = np.array([r["loss"] for r in rows])
    cls = np.array([r["loss_rpn_cls"] + sum(r[f"s{i}.loss_cls"] for i in range(3)) for r in rows])
    box = np.array([r["loss_rpn_bbox"] + sum(r[f"s{i}.loss_bbox"] for i in range(3)) for r in rows])
    lr = np.array([r["lr"] for r in rows])
    n = max(3, len(x) // 80)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), facecolor=SURFACE, sharex=True, gridspec_kw=dict(height_ratios=(3, 3, 1.4)))
    axes[0].plot(x, loss, color=SERIES[0], alpha=0.25, linewidth=0.7, zorder=2)
    axes[0].plot(x, smooth(loss, n), color=SERIES[0], linewidth=2, zorder=3, label="total (smoothed)")
    style(axes[0], "Total loss")
    axes[0].set_ylim(0, None)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2)
    for i, (name, y) in enumerate((("classification (RPN + 3 stages)", cls), ("box regression (RPN + 3 stages)", box))):
        axes[1].plot(x, smooth(y, n), color=SERIES[i], linewidth=2, label=name, zorder=3)
    style(axes[1], "Loss terms (smoothed)")
    axes[1].set_ylim(0, None)
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK_2)
    axes[2].plot(x, lr, color=INK_2, linewidth=1.5)
    style(axes[2], "Learning rate")
    finish(fig, axes, "Cascade R-CNN training loss, 4 epochs on aiMotive train at 1.5x image size (COCO-pretrained)",
           HERE / "figures" / "loss_cascade.png")


def fusion_runs(name):
    """All seeds of a configuration found in the result files, first source wins per seed."""
    runs = {}
    for src in FUSION_SOURCES:
        path = HERE / src
        if not path.exists():
            continue
        entry = json.load(open(path)).get(name)
        for r in (entry or {}).get("runs", []):
            if r.get("loss") and r["seed"] not in runs:
                runs[r["seed"]] = r["loss"]
    return dict(sorted(runs.items()))


def fusion(names):
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), facecolor=SURFACE, sharex=True, gridspec_kw=dict(height_ratios=(3, 1.2)))
    steps_per_epoch = None
    for i, name in enumerate(names):
        runs = fusion_runs(name)
        if not runs:
            print("no loss logs for", name)
            continue
        logs = list(runs.values())
        steps = np.array([p["step"] for p in logs[0]])
        steps_per_epoch = steps_per_epoch or (steps[-1] + 200) / 3
        x = steps / steps_per_epoch
        for log in logs:
            axes[0].plot(x[:len(log)], smooth([p["loss"] for p in log], 9)[:len(x)], color=SERIES[i], alpha=0.3, linewidth=0.8, zorder=2)
        n = min(len(log) for log in logs)
        mean = np.mean([[p["loss"] for p in log[:n]] for log in logs], 0)
        axes[0].plot(x[:n], smooth(mean, 9), color=SERIES[i], linewidth=2.2, zorder=3, label=f"{name} (mean of {len(logs)} seeds)")
        if i == 0:
            axes[1].plot(x, [p["lr"] for p in logs[0]], color=INK_2, linewidth=1.5)
    style(axes[0], "Focal loss per positive (smoothed)")
    axes[0].set_ylim(0, None)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2)
    style(axes[1], "Learning rate")
    finish(fig, axes, "Fusion network training loss (thin lines: individual seeds; range-weighted, far weight 2)",
           HERE / "figures" / "loss_fusion.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", default=None, help="comma-separated configurations, at most 3 (default: the final pair)")
    args = ap.parse_args()
    if args.fusion:
        names = args.fusion.split(",")
    elif (HERE / "ablation" / "decisions.json").exists():
        final = json.load(open(HERE / "ablation" / "decisions.json"))["final"]
        names = [final["clocs_r"], final["clocs"]]
    else:
        names = ["R2+T2", "R0+T2"]
    assert len(names) <= 3, "at most 3 series per plot"
    second()
    cascade()
    fusion(names)


if __name__ == "__main__":
    main()
