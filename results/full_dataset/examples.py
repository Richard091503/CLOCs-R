"""Example val frames where the models succeed and fail, for EXAMPLES.md.

    python results/full_dataset/examples.py [--clocs_r R2+T2 --clocs R0+T2 --weights runs/fusion_full --gate 0.5,0.02]

Defaults come from ablation/decisions.json (the final configurations, seed 1). Each model gets one operating threshold,
the score that maximizes F1 on holdout (a detection is correct within 2 m of a car's center, matched greedily by score).
Frames are then picked from val by category, at most one per scene and category, preferring the clearest case:

  successes: radar recovers a far car CLOCs misses; radar removes a far false alarm CLOCs keeps; fusion recovers a car SECOND misses
  failures:  fusion drops a far car SECOND found; a far car no model finds; a far false alarm radar adds; a car missed at night or in rain

Each figure shows the camera image with the three models' boxes, a zoom on the object, a bird's-eye view around it with
LiDAR and radar returns, and the whole 0-200 m scene from above. Writes figures/examples/*.png and EXAMPLES.md.
"""
import argparse
import collections
import json
import pathlib
import pickle

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.image import imread

from clocs_r import aimotive, cache_cells, fusion, radar_features, train_fusion, tune_fusion

HERE = pathlib.Path(__file__).parent
FIG = HERE / "figures" / "examples"
MATCH_M = 2.0
CLEAR_M = 4.0  # a "miss" (or a removed false alarm) needs no detection above threshold within this distance, not just a poor fit
FAR = 50.0
PER_CATEGORY = 2
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODEL_COLORS = {"SECOND": "#2a78d6", "CLOCs": "#eb6834", "CLOCs-R": "#1baf7a"}
GT_COLOR = "#0b0b0b"
RADAR_COLOR = "#6250d6"

CATEGORIES = [
    ("radar_recovers", "Success: radar recovers a far car", "A car beyond 50 m with at least one radar return in its gate that CLOCs-R detects and CLOCs, the same fusion without radar, misses (no CLOCs detection above its threshold within 4 m)."),
    ("radar_rejects", "Success: radar removes a far false alarm", "A detection beyond 50 m with no car within 2 m that CLOCs keeps and CLOCs-R drops (no CLOCs-R detection within 4 m)."),
    ("fusion_recovers", "Success: fusion recovers a car SECOND misses", "A car that both fusion models detect and SECOND alone misses (no SECOND detection within 4 m)."),
    ("fusion_drops", "Failure: fusion drops a far car SECOND found", "A car beyond 50 m that SECOND detects and neither fusion model does (no fusion detection above threshold within 4 m)."),
    ("nobody_finds", "Failure: a far car no model finds", "A car beyond 50 m that no model detects (nothing above threshold within 4 m). The caption gives its LiDAR and radar evidence."),
    ("radar_false_alarm", "Failure: radar adds a far false alarm", "A detection beyond 50 m with no car within 2 m and at least one radar return in its gate that CLOCs-R keeps and CLOCs does not (no CLOCs detection within 4 m)."),
    ("adverse_miss", "Failure: a car missed at night or in rain", "A car within 100 m in a night or rain scene that CLOCs-R misses (nothing above threshold within 4 m)."),
]


def match(gt, boxes, scores):
    """Greedy matching by descending score within MATCH_M of a car's center: per detection the matched car (or -1)."""
    order = np.argsort(-scores)
    taken = np.zeros(len(gt), bool)
    out = np.full(len(boxes), -1)
    if len(gt) == 0:
        return out
    for j in order:
        d = np.hypot(gt[:, 0] - boxes[j, 0], gt[:, 1] - boxes[j, 1])
        d[taken] = np.inf
        k = int(np.argmin(d))
        if d[k] <= MATCH_M:
            taken[k] = True
            out[j] = k
    return out


def f1_threshold(preds):
    """Score threshold with the best F1 over a split."""
    rows, n_gt = [], 0
    for item, boxes, scores in preds:
        n_gt += len(item["gt"])
        m = match(item["gt"], boxes, scores)
        rows += list(zip(scores, m >= 0))
    rows.sort(key=lambda r: -r[0])
    s = np.array([r[0] for r in rows])
    tp = np.cumsum([r[1] for r in rows])
    f1 = 2 * tp / (np.arange(1, len(rows) + 1) + n_gt)
    i = int(np.argmax(f1))
    return float(s[i]), float(f1[i])


def models(args):
    device = torch.device("cuda")
    nets = {"SECOND": (None, [])}
    for label, name in (("CLOCs", args.clocs), ("CLOCs-R", args.clocs_r)):
        cfg = tune_fusion.parse(name)
        idx = tune_fusion.names_idx(cfg)
        net = fusion.FusionNet(len(idx), **cfg["net"]).to(device)
        state = torch.load(pathlib.Path(args.weights) / f"{name}_s{args.seed}.pth", map_location="cpu")
        net.load_state_dict(state["model"])
        nets[label] = (net, idx, cfg)
    return nets, device


def predictions(split, nets, device, args):
    base, _ = cache_cells.load(split, args.cells, *args.gate_t, cache_cells.SWEEPS, use_radar=False)
    out = {}
    for label, spec in nets.items():
        net, idx = spec[0], spec[1]
        radar = [None] * len(base)
        if idx:
            cfg = spec[2]
            radar = cache_cells.load_radar(split, args.cells, *args.gate_t, cfg["sweeps"], cfg["motion"])
            if cfg["rcs"]:
                path = pathlib.Path(args.cells) / f"{split}_rcs_{cache_cells.radar_name(*args.gate_t, cfg['sweeps'], cfg['motion'])}.pkl"
                radar = np.concatenate([radar, pickle.load(open(path, "rb"))[..., None]], -1)
        out[label] = [(it, b, s) for it, b, s in train_fusion.predict(net, base, radar, idx, device)]
    return out


def frame_outcomes(preds, thresholds):
    """Per val frame and model: kept boxes and scores above threshold, matched cars and false alarms."""
    frames = []
    for i in range(len(preds["SECOND"])):
        item = preds["SECOND"][i][0]
        gt = item["gt"]
        f = dict(item=item, gt=gt, models={})
        for label in MODEL_COLORS:
            _, boxes, scores = preds[label][i]
            keep = scores >= thresholds[label]
            b, s = boxes[keep], scores[keep]
            m = match(gt, b, s)
            found = np.zeros(len(gt), bool)
            found[m[m >= 0]] = True
            near = np.array([len(b) > 0 and np.hypot(b[:, 0] - g[0], b[:, 1] - g[1]).min() <= CLEAR_M for g in gt], bool)
            f["models"][label] = dict(boxes=b, scores=s, found=found, near=near, fp=b[m < 0], fp_scores=s[m < 0], all_boxes=boxes, all_scores=scores)
        frames.append(f)
    return frames


def best_score_near(model, xy):
    """Highest score of any of the model's detections (before the threshold) within MATCH_M of xy."""
    d = np.hypot(model["all_boxes"][:, 0] - xy[0], model["all_boxes"][:, 1] - xy[1]) if len(model["all_boxes"]) else np.zeros(0)
    near = d <= MATCH_M
    return float(model["all_scores"][near].max()) if near.any() else None


def candidates(frames, thresholds, radar):
    """For each category, (strength, frame index, object xy, kind) of every qualifying object.

    radar: CLOCs-R's cached radar features per val frame and SECOND candidate; the radar categories need at least one
    return in the gate of the candidate at the object (otherwise a difference comes from training, not from radar evidence).
    """
    out = collections.defaultdict(list)

    def has_return(fi, xy):
        if radar is None:
            return False
        boxes = frames[fi]["item"]["boxes"]
        d = np.hypot(boxes[:, 0] - xy[0], boxes[:, 1] - xy[1])
        j = int(np.argmin(d))
        return bool(d[j] <= MATCH_M and radar[fi][j, 0] > 0)

    for fi, f in enumerate(frames):
        gt, M = f["gt"], f["models"]
        dist = np.hypot(gt[:, 0], gt[:, 1]) if len(gt) else np.zeros(0)
        for k in range(len(gt)):
            sec, clo, cr = (M[x]["found"][k] for x in MODEL_COLORS)
            miss_sec, miss_clo, miss_cr = (not M[x]["near"][k] for x in MODEL_COLORS)
            xy = gt[k, :2]
            if dist[k] >= FAR and cr and miss_clo and has_return(fi, xy):
                s = best_score_near(M["CLOCs"], xy) or 0.0
                out["radar_recovers"].append((thresholds["CLOCs"] - s, fi, xy, "car"))
            if (clo and cr) and miss_sec:
                out["fusion_recovers"].append((dist[k], fi, xy, "car"))
            if dist[k] >= FAR and sec and miss_clo and miss_cr:
                out["fusion_drops"].append((dist[k], fi, xy, "car"))
            if dist[k] >= FAR and miss_sec and miss_clo and miss_cr:
                out["nobody_finds"].append((-abs(dist[k] - 90), fi, xy, "car"))  # prefer typical far range over the edge
            if f["item"]["odd"] in ("night", "rain") and dist[k] < 100 and miss_cr:
                out["adverse_miss"].append((-abs(dist[k] - 60), fi, xy, "car"))  # prefer the middle of the range over the edges
        for a, b, cat in (("CLOCs", "CLOCs-R", "radar_rejects"), ("CLOCs-R", "CLOCs", "radar_false_alarm")):
            for box, s in zip(M[a]["fp"], M[a]["fp_scores"]):
                if np.hypot(box[0], box[1]) < FAR or (cat == "radar_false_alarm" and not has_return(fi, box[:2])):
                    continue
                kept = M[b]["boxes"]
                if len(kept) == 0 or np.hypot(kept[:, 0] - box[0], kept[:, 1] - box[1]).min() > CLEAR_M:
                    out[cat].append((s - thresholds[a], fi, box[:2], "false alarm"))
    return out


def pick(cands, frames):
    """Strongest cases first, one per scene, covering as many driving conditions as possible."""
    chosen = {}
    for key, _, _ in CATEGORIES:
        rows = sorted(cands.get(key, []), key=lambda r: -r[0])
        picked, scenes, odds = [], set(), set()
        for new_condition_only in (True, False):
            for strength, fi, xy, kind in rows:
                item = frames[fi]["item"]
                if len(picked) == PER_CATEGORY or item["seq"] in scenes or (new_condition_only and item["odd"] in odds):
                    continue
                picked.append((fi, xy, kind))
                scenes.add(item["seq"])
                odds.add(item["odd"])
        chosen[key] = picked
    return chosen


def object_evidence(seq, frame, gt, xy, sweeps):
    pts = seq.lidar(frame)
    rpts, rvel = seq.radar_accumulated(frame, sweeps)
    k = int(np.argmin(np.hypot(gt[:, 0] - xy[0], gt[:, 1] - xy[1]))) if len(gt) else -1
    info = dict(distance=float(np.hypot(*xy)))
    if k >= 0 and np.hypot(*(gt[k, :2] - xy)) <= MATCH_M:
        box = gt[k:k + 1]
        dx, dy = pts[:, 0] - box[0, 0], pts[:, 1] - box[0, 1]
        c, s = np.cos(box[0, 6]), np.sin(box[0, 6])
        inside = (np.abs(dx * c + dy * s) <= box[0, 3] / 2) & (np.abs(-dx * s + dy * c) <= box[0, 4] / 2) & (np.abs(pts[:, 2] - box[0, 2]) <= box[0, 5] / 2)
        info["lidar_points"] = int(inside.sum())
        f = radar_features.candidate_features(box, rpts, rvel, *tune_fusion.GATE, names=("r", "v"))
        info["radar_returns"] = int(round(np.expm1(f[0, 0])))
        info["doppler"] = float(f[0, 1] * radar_features.V_MAX) if info["radar_returns"] else None
        uv, zc = seq.project(aimotive.box_corners_3d(box.astype(np.float64))[0])
        w, h = seq.meta["image_size"]
        info["cut_off"] = bool((zc <= 0.1).any() or uv[:, 0].min() < 0 or uv[:, 0].max() > w)  # extends past the image's side
    return pts, rpts, rvel, info


def draw(f, xy, kind, key, title, idx, thresholds, args):
    item = f["item"]
    seq = aimotive.Sequence(next(p for p in aimotive.list_sequences(aimotive.DEFAULT_ROOT) if p.endswith(item["seq"])))
    frame = item["frame"]
    pts, rpts, rvel, info = object_evidence(seq, frame, f["gt"], xy, args.sweeps)
    _, _, det2d = cache_cells.load_candidates(seq, frame, args.cand3d, args.cand2d)
    img = imread(seq.image_path(frame))
    H, W = img.shape[:2]

    fig = plt.figure(figsize=(12, 10.5), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=(5.2, 3.2, 1.25), width_ratios=(1, 1), hspace=0.25, wspace=0.12)
    ax_img, ax_zoom, ax_bev, ax_strip = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, :])

    def boxes2d(boxes):
        rect, valid = aimotive.project_boxes(seq, boxes) if len(boxes) else (np.zeros((0, 4)), np.zeros(0, bool))
        return rect[valid]

    uv, z = seq.project(np.array([[xy[0], xy[1], f["gt"][:, 2].mean() if len(f["gt"]) else 0.0]]))
    cu, cv = float(np.clip(uv[0, 0], 0, W)), float(np.clip(uv[0, 1], 0, H))
    half = 60 if info["distance"] > 80 else 110
    zoom = (max(0, cu - 1.6 * half), min(W, cu + 1.6 * half), min(H, cv + half), max(0, cv - half))

    for ax in (ax_img, ax_zoom):
        ax.imshow(img)
        ax.axis("off")
        for r in det2d[det2d[:, 4] >= 0.3]:
            ax.add_patch(mpatches.Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fill=False, ec="#ffffff", lw=0.6, ls=":", alpha=0.9))
        for r in boxes2d(f["gt"]):
            ax.add_patch(mpatches.Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fill=False, ec="#ffffff", lw=1.6))
            ax.add_patch(mpatches.Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fill=False, ec=GT_COLOR, lw=0.8, ls="--"))
        for i, (label, color) in enumerate(MODEL_COLORS.items()):
            for r in boxes2d(f["models"][label]["boxes"]):
                pad = 1.5 * (i + 1)
                ax.add_patch(mpatches.Rectangle((r[0] - pad, r[1] - pad), r[2] - r[0] + 2 * pad, r[3] - r[1] + 2 * pad, fill=False, ec=color, lw=1.4))
    ax_img.add_patch(mpatches.Rectangle((zoom[0], zoom[3]), zoom[1] - zoom[0], zoom[2] - zoom[3], fill=False, ec="#eda100", lw=1.8))
    ax_img.set_title(f"{item['odd'].capitalize()} | {item['seq']} | frame {frame}", color=INK, fontsize=9.5, loc="left")
    ax_zoom.set_xlim(zoom[0], zoom[1])
    ax_zoom.set_ylim(zoom[2], zoom[3])
    ax_zoom.set_title("Camera, zoomed (yellow frame above)", color=INK_2, fontsize=9, loc="left")

    def bev(ax, xlim, ylim, point_size, label_scores):
        ax.set_facecolor(SURFACE)
        sel = (pts[:, 0] > xlim[0]) & (pts[:, 0] < xlim[1]) & (pts[:, 1] > ylim[0]) & (pts[:, 1] < ylim[1])
        ax.scatter(pts[sel, 0], pts[sel, 1], s=point_size, c=MUTED, linewidths=0, zorder=1)
        rs = (rpts[:, 0] > xlim[0]) & (rpts[:, 0] < xlim[1]) & (rpts[:, 1] > ylim[0]) & (rpts[:, 1] < ylim[1])
        ax.scatter(rpts[rs, 0], rpts[rs, 1], s=point_size * 14 + 6, marker="^", c=RADAR_COLOR, linewidths=0, zorder=4)
        for c in aimotive.box_corners_bev(f["gt"]) if len(f["gt"]) else []:
            ax.add_patch(mpatches.Polygon(c, closed=True, fill=False, ec=GT_COLOR, lw=1.2, ls="--", zorder=3))
        for label, color in MODEL_COLORS.items():
            m = f["models"][label]
            for c, s in zip(aimotive.box_corners_bev(m["boxes"]) if len(m["boxes"]) else [], m["scores"]):
                ax.add_patch(mpatches.Polygon(c, closed=True, fill=False, ec=color, lw=1.6, zorder=5))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)  # y (left of the car) points up, x forward to the right
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)

    bev(ax_bev, (xy[0] - 18, xy[0] + 18), (xy[1] - 9, xy[1] + 9), 3.0, True)
    ax_bev.set_aspect("equal")
    ax_bev.add_patch(mpatches.Circle(xy, MATCH_M, fill=False, ec="#eda100", lw=1.5, zorder=6))
    ax_bev.set_title("Bird's-eye view around the object (x forward, m; circle: 2 m match radius)", color=INK_2, fontsize=9, loc="left")
    bev(ax_strip, (0, 205), (-25.6, 25.6), 0.4, False)
    ax_strip.add_patch(mpatches.Circle(xy, 4, fill=False, ec="#eda100", lw=1.5, zorder=6))
    ax_strip.axvline(FAR, color=INK_2, lw=0.8, ls="--", zorder=0)
    ax_strip.set_title("Whole scene from above, 0-205 m (dashed: 50 m)", color=INK_2, fontsize=9, loc="left")

    legend = [mpatches.Patch(fc="none", ec=GT_COLOR, ls="--", label="labelled car"), mpatches.Patch(fc="none", ec="#ffffff", ls=":", label="Cascade R-CNN 2D box (score ≥ 0.3)")] + \
             [mpatches.Patch(fc="none", ec=c, lw=1.6, label=f"{m} (score ≥ {thresholds[m]:.3f})") for m, c in MODEL_COLORS.items()] + \
             [plt.Line2D([], [], marker="^", ls="", color=RADAR_COLOR, label="radar return"), plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=3, label="LiDAR point")]
    leg = fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK_2)
    handles = getattr(leg, "legend_handles", None) or leg.legendHandles
    handles[1].set_edgecolor(INK_2)  # the white 2D-box style is invisible on the light legend background
    scores = {m: best_score_near(f["models"][m], xy) for m in MODEL_COLORS}
    fig.suptitle(title, color=INK, fontsize=11.5, x=0.01, ha="left", y=0.995)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.08)
    name = f"{key}_{idx + 1}.png"
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return dict(file=f"figures/examples/{name}", scene=item["seq"], odd=item["odd"], frame=int(frame), kind=kind, scores=scores, **info)


def fmt_score(s, th):
    return "none within 2 m" if s is None else f"{s:.3f} ({'above' if s >= th else 'below'} {th:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clocs_r", default=None)
    ap.add_argument("--clocs", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--gate", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cells", default="runs/cells")
    ap.add_argument("--cand3d", default="runs/candidates3d")
    ap.add_argument("--cand2d", default="runs/candidates2d")
    args = ap.parse_args()
    dec_path = HERE / "ablation" / "decisions.json"
    final = json.load(open(dec_path))["final"] if dec_path.exists() else dict(clocs_r="R2+T2", clocs="R0+T2", gate=[0.5, 0.02])
    args.clocs_r = args.clocs_r or final["clocs_r"]
    args.clocs = args.clocs or final["clocs"]
    args.gate_t = tuple(float(x) for x in args.gate.split(",")) if args.gate else tuple(final["gate"])
    tune_fusion.GATE = args.gate_t
    args.weights = args.weights or (f"runs/ablation/sets_d{args.gate_t[0]:g}_k{args.gate_t[1]:g}" if dec_path.exists() else "runs/fusion_full")
    args.sweeps = tune_fusion.parse(args.clocs_r)["sweeps"]
    nets, device = models(args)

    cache = pathlib.Path("runs/examples") / f"{args.clocs_r}_{args.clocs}_s{args.seed}_d{args.gate_t[0]:g}_k{args.gate_t[1]:g}.pkl"
    if cache.exists():  # predictions are deterministic; this only saves the 15 minutes of scoring holdout and val again
        thresholds, f1, val = pickle.load(open(cache, "rb"))
        base, _ = cache_cells.load("val", args.cells, *args.gate_t, cache_cells.SWEEPS, use_radar=False)
        preds = {m: [(it, b, s) for it, (b, s) in zip(base, val[m])] for m in MODEL_COLORS}
    else:
        thresholds, f1 = {}, {}
        hold = predictions("holdout", nets, device, args)
        for label in MODEL_COLORS:
            thresholds[label], f1[label] = f1_threshold(hold[label])
        del hold
        preds = predictions("val", nets, device, args)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((thresholds, f1, {m: [(b, s) for _, b, s in preds[m]] for m in MODEL_COLORS}), open(cache, "wb"))
    print("thresholds:", thresholds, "holdout F1:", f1, flush=True)
    frames = frame_outcomes(preds, thresholds)
    cfg = tune_fusion.parse(args.clocs_r)
    radar = cache_cells.load_radar("val", args.cells, *args.gate_t, cfg["sweeps"], cfg["motion"]) if cfg["radar"] else None
    cands = candidates(frames, thresholds, radar)
    chosen = pick(cands, frames)

    md = ["# Example frames: where the models succeed and fail", "",
          f"Val frames picked automatically by `examples.py`, comparing SECOND alone, CLOCs ({args.clocs}, no radar) and CLOCs-R "
          f"({args.clocs_r}), seed {args.seed}. Each model runs at one fixed score threshold, the one that maximizes F1 on holdout: "
          + ", ".join(f"{m} {thresholds[m]:.3f} (F1 {f1[m]:.3f})" for m in MODEL_COLORS) + ". A detection counts when it lies within "
          "2 m of a labelled car's center. Within a category, frames come from different scenes, and the clearest cases come first "
          "(for example the largest score margin). They illustrate typical mechanisms and are not a random sample.", "",
          "Camera images, LiDAR and radar data and labels in these figures are from the aiMotive multimodal dataset, used under CC BY-NC-SA 4.0. The figures are shared under the same license.", "",
          "In each figure:", "",
          "- **Top:** the camera image. The dashed boxes are labelled cars and the dotted white boxes are Cascade R-CNN's 2D detections. "
          "Colored boxes are each model's detections above its threshold (blue SECOND, orange CLOCs, green CLOCs-R).",
          "- **Middle left:** a zoom on the yellow frame.",
          "- **Middle right:** a bird's-eye view around the object, with LiDAR points (gray) and radar returns from the last "
          f"{args.sweeps} sweeps (violet triangles).",
          "- **Bottom:** the whole scene from above, out to 205 m.", ""]
    report = {}
    for key, title, explain in CATEGORIES:
        md += [f"## {title}", "", explain, ""]
        report[key] = []
        if not chosen[key]:
            md += ["No frame in val fits this category at these thresholds.", ""]
        for i, (fi, xy, kind) in enumerate(chosen[key]):
            r = draw(frames[fi], xy, kind, key, title, i, thresholds, args)
            report[key].append(r)
            ev = []
            if "lidar_points" in r:
                ev.append(f"{r['lidar_points']} LiDAR points in the box, {r['radar_returns']} radar return{'s' if r['radar_returns'] != 1 else ''} in the gate"
                          + (f" (median Doppler speed {r['doppler']:.1f} m/s)" if r.get("doppler") is not None else "")
                          + (". The car extends past the side of the image, and LiDAR is kept only inside the camera's view, so "
                             "the detectors see part of it" if r.get("cut_off") else ""))
            md += [f"![{title} {i + 1}]({r['file']})", "",
                   f"**{r['odd'].capitalize()}**, scene `{r['scene']}`, frame {r['frame']}: {r['kind']} at {r['distance']:.0f} m. "
                   + "Best score within 2 m: " + "; ".join(f"{m} {fmt_score(r['scores'][m], thresholds[m])}" for m in MODEL_COLORS) + ". "
                   + (ev[0] + "." if ev else ""), ""]
        print(key, len(chosen[key]), "examples", flush=True)
    counts = {k: len(v) for k, v in cands.items()}
    md += ["## How common each case is on val", "",
           "Objects that meet each category's conditions, over all val frames, at the thresholds above:", "",
           "| Category | Objects |", "|---|---|"] + [f"| {t} | {counts.get(k, 0):,} |" for k, t, _ in CATEGORIES]
    (HERE / "EXAMPLES.md").write_text("\n".join(md) + "\n")
    json.dump(dict(models=dict(clocs_r=args.clocs_r, clocs=args.clocs, seed=args.seed), thresholds=thresholds, holdout_f1=f1, counts=counts,
                   examples=report), open(HERE / "examples.json", "w"), indent=1, default=float)
    print("wrote", HERE / "EXAMPLES.md")


if __name__ == "__main__":
    main()
