"""Build the metrics, plots and sample figures for a CLOCs training run.

Usage (from the CLOCs root, with env.sh sourced):
    python results/make_report.py --run_dir training_runs/car_short_2epoch

Reads <run_dir>/log.txt and <run_dir>/results/step_*/result.pkl, plus the
pretrained-model eval log for comparison, and writes into results/:
    metrics.csv, metrics.md, loss.csv, figures/*.png, samples/*.txt
"""
import argparse
import csv
import pathlib
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from second.core import box_np_ops

ROOT = pathlib.Path(__file__).resolve().parents[1]
KITTI = pathlib.Path.home() / "Documents" / "KITTI_DATASET_ROOT"
OUT = ROOT / "results"
SAMPLE_IDS = [39, 152, 186, 345, 370, 408, 455, 548]
SCORE_THRESH = 0.5  # only detections at or above this score are drawn

# Validated categorical palette (dataviz reference instance), light mode
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PRED, GT, PRETRAINED = "#2a78d6", "#eb6834", "#1baf7a"
DIFFS = ["Easy", "Moderate", "Hard"]
METRICS = ["bbox", "bev", "3d", "aos"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def parse_ap_tables(text):
    """Every 'Car AP@0.70, 0.70, 0.70' block, in log order -> list of {metric: [E, M, H]}."""
    tables = []
    for m in re.finditer(r"Car AP@0\.70, 0\.70, 0\.70:\n((?:.*\n){4})", text):
        t = {}
        for line in m.group(1).splitlines():
            name, vals = line.split("AP:")
            t[name.strip()] = [float(v) for v in vals.split(",")]
        tables.append(t)
    return tables


def parse_losses(text):
    pat = re.compile(r"now it is (\d+) steps\s+and the cls_loss is :\s*(?:tensor\()?([-0-9.eE+]+).*?learning_rate:\s*([-0-9.eE+]+)")
    return [(int(s), float(l), float(lr)) for s, l, lr in pat.findall(text)]


def write_metrics(run_dir, run_tables, eval_steps, pretrained):
    rows = []
    for step, t in zip(eval_steps, run_tables):
        for metric in METRICS:
            rows.append([f"short run, step {step}", metric] + t[metric])
    if pretrained:
        for metric in METRICS:
            rows.append(["authors' pretrained (step 11136)", metric] + pretrained[metric])
    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "AP_easy", "AP_moderate", "AP_hard"])
        w.writerows(rows)
    lines = ["| Model | Metric | Easy | Moderate | Hard |", "|---|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} |" for r in rows]
    (OUT / "metrics.md").write_text("\n".join(lines) + "\n")


def plot_loss(losses):
    steps, loss, lr = map(np.array, zip(*losses))
    with open(OUT / "loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "cls_loss_avg_last_50_steps", "learning_rate"])
        w.writerows(zip(steps, loss, lr))
    # Two measures with different scales -> two stacked charts, never a dual axis
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]}, facecolor=SURFACE)
    for ax in (a1, a2):
        style(ax)
    a1.plot(steps, loss, color=PRED, linewidth=2)
    a1.set_yscale("log")
    a1.set_title("CLOCs fusion training loss (focal loss, mean of each 50 steps)",
                 color=INK, fontsize=11, loc="left")
    a1.set_ylabel("Loss (log scale)", color=INK_2)
    a2.plot(steps, lr, color=INK_2, linewidth=2)
    a2.set_title("Learning rate (one-cycle schedule)", color=INK, fontsize=11, loc="left")
    a2.set_xlabel("Training step", color=INK_2)
    a2.set_ylabel("LR", color=INK_2)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "training_loss.png", dpi=150)
    plt.close(fig)


def plot_ap(run_tables, eval_steps, pretrained):
    epoch_colors = [PRED, GT]  # slots 1-2 for this run's epochs; slot 3 always = pretrained
    series = [(f"Short run, epoch {i + 1} (step {s})", t["3d"], epoch_colors[i])
              for i, (s, t) in enumerate(zip(eval_steps[-2:], run_tables[-2:]))]
    if pretrained:
        series.append(("Authors' pretrained CLOCs", pretrained["3d"], PRETRAINED))
    x = np.arange(len(DIFFS))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style(ax)
    for i, (name, vals, c) in enumerate(series):
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width - 0.02,
                      color=c, label=name, edgecolor=SURFACE, linewidth=2)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=INK_2, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(DIFFS, color=INK)
    ax.set_ylim(0, 105)
    ax.set_ylabel("3D AP (%), IoU 0.7, 40 recall points", color=INK_2)
    ax.set_title("Car 3D detection AP on KITTI val (3,769 frames)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center", ncol=len(series), bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "ap_3d_by_epoch.png", dpi=150)
    plt.close(fig)


def cam_corners(loc, dims, rot):
    return box_np_ops.center_to_corner_box3d(loc, dims, rot, origin=[0.5, 1.0, 0.5], axis=1)


def project(corners, P2):
    pts = np.concatenate([corners, np.ones(corners.shape[:-1] + (1,))], axis=-1) @ P2.T
    return pts[..., :2] / pts[..., 2:3]


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_sample(info, det, step):
    idx = info["image_idx"]
    P2, rect, Trv2c = info["calib/P2"], info["calib/R0_rect"], info["calib/Tr_velo_to_cam"]
    gt = info["annos"]
    gm = gt["name"] == "Car"
    keep = det["score"] >= SCORE_THRESH

    fig = plt.figure(figsize=(12, 10), facecolor=SURFACE)
    ax_img = fig.add_axes([0.02, 0.6, 0.96, 0.36])
    ax_bev = fig.add_axes([0.08, 0.07, 0.86, 0.46])

    # Camera view with projected 3D boxes
    img = np.asarray(Image.open(KITTI / "training" / "image_2" / f"{idx:06d}.png"))
    ax_img.imshow(img)
    ax_img.axis("off")
    for loc, dims, rot, color, lw in [(gt["location"][gm], gt["dimensions"][gm], gt["rotation_y"][gm], GT, 2.5),
                                      (det["location"][keep], det["dimensions"][keep], det["rotation_y"][keep], PRED, 1.5)]:
        if len(loc) == 0:
            continue
        c2d = project(cam_corners(loc, dims, rot), P2)
        for box in c2d:
            for a, b in EDGES:
                ax_img.plot(box[[a, b], 0], box[[a, b], 1], color=color, linewidth=lw)
    ax_img.set_xlim(0, img.shape[1])
    ax_img.set_ylim(img.shape[0], 0)
    ax_img.set_title(f"KITTI val frame {idx:06d}: ground truth (orange) vs CLOCs predictions with score >= {SCORE_THRESH} (blue)",
                     color=INK, fontsize=11, loc="left")

    # Bird's-eye view: LiDAR points (front-camera field of view) and boxes
    pts = np.fromfile(KITTI / "training" / "velodyne_reduced" / f"{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)
    ax_bev.set_facecolor(SURFACE)
    ax_bev.scatter(pts[:, 0], pts[:, 1], s=0.2, color=INK_2, alpha=0.5, linewidths=0)
    for src, m, color, lw in [(gt, gm, GT, 2.5), (det, keep, PRED, 1.5)]:
        if m.sum() == 0:
            continue
        cam = np.concatenate([src["location"][m], src["dimensions"][m], src["rotation_y"][m][:, None]], axis=1)
        lidar = box_np_ops.box_camera_to_lidar(cam, rect, Trv2c)
        for poly in box_np_ops.center_to_corner_box2d(lidar[:, :2], lidar[:, 3:5], lidar[:, 6]):
            ring = np.vstack([poly, poly[:1]])
            ax_bev.plot(ring[:, 0], ring[:, 1], color=color, linewidth=lw)
    ax_bev.set_xlim(0, 70.4)
    ax_bev.set_ylim(-25, 25)
    ax_bev.set_aspect("equal")
    ax_bev.invert_yaxis()  # left of the car at the top, as seen from above facing forward
    ax_bev.set_xlabel("Forward distance from LiDAR (m)", color=INK_2)
    ax_bev.set_ylabel("Lateral (m)", color=INK_2)
    ax_bev.tick_params(colors=INK_2, labelsize=8)
    for s in ax_bev.spines.values():
        s.set_color(GRID)
    ax_bev.set_title("Bird's-eye view (LiDAR points inside the camera view)", color=INK, fontsize=10, loc="left")
    ax_bev.plot([], [], color=GT, linewidth=2.5, label="Ground truth (Car)")
    ax_bev.plot([], [], color=PRED, linewidth=1.5, label=f"CLOCs prediction, score >= {SCORE_THRESH}")
    ax_bev.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.savefig(OUT / "figures" / f"sample_{idx:06d}.png", dpi=110)
    plt.close(fig)

    # Text version of the same output
    lines = [f"KITTI val frame {idx:06d} - CLOCs short run, step {step}",
             f"Ground-truth cars: {int(gm.sum())}   Predictions with score >= {SCORE_THRESH}: {int(keep.sum())}",
             "", "Predictions (camera coords: x right, y down, z forward; metres)",
             f"{'score':>6} {'x':>7} {'y':>6} {'z':>7} {'l':>5} {'h':>5} {'w':>5} {'yaw':>6}"]
    order = np.argsort(-det["score"])
    for i in order[keep[order]]:
        x, y, z = det["location"][i]
        l, h, w = det["dimensions"][i]
        lines.append(f"{det['score'][i]:6.3f} {x:7.2f} {y:6.2f} {z:7.2f} {l:5.2f} {h:5.2f} {w:5.2f} {det['rotation_y'][i]:6.2f}")
    lines += ["", "Ground truth cars", f"{'x':>7} {'y':>6} {'z':>7} {'l':>5} {'h':>5} {'w':>5} {'yaw':>6}  difficulty"]
    for i in np.where(gm)[0]:
        x, y, z = gt["location"][i]
        l, h, w = gt["dimensions"][i]
        diff = {0: "easy", 1: "moderate", 2: "hard"}.get(int(gt["difficulty"][i]), "ignored") if "difficulty" in gt else ""
        lines.append(f"{x:7.2f} {y:6.2f} {z:7.2f} {l:5.2f} {h:5.2f} {w:5.2f} {gt['rotation_y'][i]:6.2f}  {diff}")
    (OUT / "samples" / f"sample_{idx:06d}.txt").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--pretrained_log", default=str(ROOT / "CLOCs_SecCas_pretrained" / "eval_log.txt"))
    args = p.parse_args()
    run_dir = pathlib.Path(args.run_dir).resolve()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(parents=True, exist_ok=True)

    log = (run_dir / "log.txt").read_text()
    run_tables = parse_ap_tables(log)
    eval_steps = sorted(int(d.name.split("_")[1]) for d in (run_dir / "results").glob("step_*"))[:len(run_tables)]
    pre_path = pathlib.Path(args.pretrained_log)
    pre_tables = parse_ap_tables(pre_path.read_text().replace("\r", "\n")) if pre_path.exists() else []
    pretrained = pre_tables[-1] if pre_tables else None

    write_metrics(run_dir, run_tables, eval_steps, pretrained)
    plot_loss(parse_losses(log))
    plot_ap(run_tables, eval_steps, pretrained)

    last = eval_steps[-1]
    dets = pickle.load(open(run_dir / "results" / f"step_{last}" / "result.pkl", "rb"))
    infos = pickle.load(open(KITTI / "kitti_infos_val.pkl", "rb"))
    by_idx = {int(i["image_idx"]): (i, d) for i, d in zip(infos, dets)}
    for idx in SAMPLE_IDS:
        info, det = by_idx[idx]
        assert int(det["image_idx"][0]) == idx if len(det["image_idx"]) else True
        draw_sample(info, det, last)
    print((OUT / "metrics.md").read_text())


if __name__ == "__main__":
    main()
