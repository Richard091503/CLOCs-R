"""Dataset analysis for DATASET.md: where the labelled vehicles are, and how much evidence each sensor has on them.

    python results/full_dataset/dataset_stats.py            # measure (about 10 min with 14 processes) and report
    python results/full_dataset/dataset_stats.py --report   # report again from the saved measurements

For every labelled object of every frame of train, holdout and val, it records the class, position, distance, the
number of LiDAR points inside the 3D box, the radar returns inside the CLOCs-R gate (box enlarged by 0.5 m + 0.02 x range,
3 sweeps, as in the fusion), their median ego-compensated Doppler speed, and the height of the projected box in the image.
Writes dataset_stats.json and dataset_tables.md here, figures/dataset_*.png, and the raw records to runs/dataset_stats/.
"""
import argparse
import collections
import json
import multiprocessing as mp
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clocs_r import aimotive, radar_features, splits

HERE = pathlib.Path(__file__).parent
RAW = pathlib.Path("runs/dataset_stats")
SPLITS = ("train", "holdout", "val")
ODDS = ("highway", "urban", "night", "rain")
BINS = ((0, 30), (30, 50), (50, 100), (100, 200))
FAR = 50.0
GATE = (0.5, 0.02)
SWEEPS = 3
SMALL_PX = 25  # the "under 25 px" bucket of the Cascade R-CNN table
MOVING = 1.0  # m/s of ego-compensated Doppler speed
# Reference palette (dataviz skill), light mode
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
FIELDS = ("split", "odd", "scene", "frame", "cls", "car", "scored", "x", "y", "dist", "lidar", "radar", "doppler", "height_px", "rel_speed")


def points_in_boxes(boxes, pts):
    """Number of points inside each 3D box (N, 7)."""
    if len(boxes) == 0 or len(pts) == 0:
        return np.zeros(len(boxes), int)
    out = np.zeros(len(boxes), int)
    for i, (x, y, z, l, w, h, yaw) in enumerate(boxes):
        near = (np.abs(pts[:, 0] - x) < l + w) & (np.abs(pts[:, 1] - y) < l + w)
        p = pts[near]
        dx, dy = p[:, 0] - x, p[:, 1] - y
        c, s = np.cos(yaw), np.sin(yaw)
        out[i] = np.count_nonzero((np.abs(dx * c + dy * s) <= l / 2) & (np.abs(-dx * s + dy * c) <= w / 2) & (np.abs(p[:, 2] - z) <= h / 2))
    return out


def measure_scene(path):
    seq = aimotive.Sequence(path)
    rows, frames = [], []
    for f in seq.kept:
        boxes, names = seq.boxes(f)
        objs = seq.meta["labels"].get(f, [])
        pts = seq.lidar(f)
        rpts, rvel = seq.radar_accumulated(f, SWEEPS)
        frames.append(dict(scene=seq.name, frame=int(f), ego_speed=float(np.linalg.norm(seq.ego_velocity(f)[:2])),
                           lidar_points=int(len(pts)), radar_returns=int(len(rpts))))
        if len(boxes) == 0:
            continue
        scored = aimotive.in_region(boxes) & aimotive.in_camera_view(seq, boxes)
        lidar = points_in_boxes(boxes, pts)
        feats = radar_features.candidate_features(boxes, rpts, rvel, *GATE, names=("r", "v"))
        n_radar = np.rint(np.expm1(feats[:, 0])).astype(int)
        doppler = np.where(n_radar > 0, feats[:, 1] * radar_features.V_MAX, np.nan)
        rect, valid = aimotive.project_boxes(seq, boxes)
        height = np.where(valid, rect[:, 3] - rect[:, 1], np.nan)
        rel = np.array([np.hypot(o["Relative Velocity X"], o["Relative Velocity Y"]) for o in objs])
        for i, name in enumerate(names):
            rows.append((None, seq.odd, seq.name, int(f), name, name in splits.CAR_CLASSES, bool(scored[i]), float(boxes[i, 0]),
                         float(boxes[i, 1]), float(np.hypot(boxes[i, 0], boxes[i, 1])), int(lidar[i]), int(n_radar[i]),
                         float(doppler[i]), float(height[i]), float(rel[i])))
    return seq.name, seq.odd, rows, frames


def measure():
    RAW.mkdir(parents=True, exist_ok=True)
    split_of = {}
    for s in SPLITS:
        for path in splits.scenes(s):
            split_of[path] = s
    with mp.Pool(14) as pool:
        results = pool.map(measure_scene, sorted(split_of), chunksize=1)
    rows, frames = [], []
    for path, (name, odd, r, fr) in zip(sorted(split_of), results):
        rows += [(split_of[path],) + row[1:] for row in r]
        frames += [dict(f, split=split_of[path], odd=odd) for f in fr]
    cols = {k: np.array([r[i] for r in rows]) for i, k in enumerate(FIELDS)}
    np.savez_compressed(RAW / "objects.npz", **cols)
    with open(RAW / "frames.json", "w") as f:
        json.dump(frames, f)
    print(f"measured {len(frames)} frames, {len(rows)} labelled objects")


def pct(x):
    return f"{100 * x:.1f}%" if np.isfinite(x) else "n/a"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def summarise(o, frames):
    """Everything DATASET.md quotes, as nested plain dicts."""
    car = o["car"] & o["scored"]
    far = (o["dist"] >= FAR) & (o["dist"] < 200)  # the evaluation's 50-200 m bin
    s = dict(frames={}, objects={}, cars={}, evidence={}, ego_speed={})
    for sp in SPLITS + ("all",):
        for odd in ODDS + ("all",):
            fm = [f for f in frames if (sp == "all" or f["split"] == sp) and (odd == "all" or f["odd"] == odd)]
            m = car & ((o["split"] == sp) if sp != "all" else True) & ((o["odd"] == odd) if odd != "all" else True)
            key = f"{sp}/{odd}"
            s["frames"][key] = dict(frames=len(fm), scenes=len({f["scene"] for f in fm}))
            s["cars"][key] = dict(total=int(m.sum()), far=int((m & far).sum()),
                                  bins={f"{a}-{b}": int((m & (o["dist"] >= a) & (o["dist"] < b)).sum()) for a, b in BINS},
                                  far_frames=int(len({(sc, fr) for sc, fr in zip(o["scene"][m & far], o["frame"][m & far])})))
    for odd in ODDS + ("all",):
        fm = [f for f in frames if odd == "all" or f["odd"] == odd]
        s["ego_speed"][odd] = dict(median=float(np.median([f["ego_speed"] for f in fm])),
                                   lidar_points=float(np.median([f["lidar_points"] for f in fm])),
                                   radar_returns=float(np.median([f["radar_returns"] for f in fm])))
    for cls in sorted(set(o["cls"])):
        m = o["cls"] == cls
        s["objects"][cls] = dict(all=int(m.sum()), scored=int((m & o["scored"]).sum()), scored_far=int((m & o["scored"] & far).sum()))
    for odd in ODDS + ("all",):
        for a, b in BINS + ((50, 200),):
            m = car & (o["dist"] >= a) & (o["dist"] < b) & ((o["odd"] == odd) if odd != "all" else True)
            n = int(m.sum())
            hit = m & (o["radar"] > 0)
            s["evidence"][f"{odd}/{a}-{b}"] = dict(
                n=n,
                lidar_any=float((o["lidar"][m] > 0).mean()) if n else np.nan,
                lidar_5=float((o["lidar"][m] >= 5).mean()) if n else np.nan,
                lidar_median=float(np.median(o["lidar"][m])) if n else np.nan,
                radar_any=float((o["radar"][m] > 0).mean()) if n else np.nan,
                radar_median_hit=float(np.median(o["radar"][hit])) if hit.any() else np.nan,
                moving=float((o["doppler"][hit] > MOVING).mean()) if hit.any() else np.nan,
                height_median=float(np.nanmedian(o["height_px"][m])) if n else np.nan,
                small=float((o["height_px"][m] < SMALL_PX).mean()) if n else np.nan,
                no_lidar_radar=float(((o["lidar"][m] == 0) & (o["radar"][m] > 0)).mean()) if n else np.nan,
                none=float(((o["lidar"][m] == 0) & (o["radar"][m] == 0)).mean()) if n else np.nan)
    return s


def report(o, frames):
    s = summarise(o, frames)
    with open(HERE / "dataset_stats.json", "w") as f:
        json.dump(s, f, indent=1, default=float)
    md = []
    rows = []
    for sp in SPLITS + ("all",):
        r = [sp]
        for odd in ODDS:
            fr, c = s["frames"][f"{sp}/{odd}"], s["cars"][f"{sp}/{odd}"]
            r.append(f"{fr['scenes']} / {fr['frames']:,}")
        fr = s["frames"][f"{sp}/all"]
        rows.append(r + [f"{fr['scenes']} / {fr['frames']:,}"])
    md.append("## Scenes / frames by split and condition\n\n" + table(["Split"] + list(ODDS) + ["All"], rows))

    rows = []
    for sp in SPLITS + ("all",):
        for odd in ODDS + ("all",):
            c = s["cars"][f"{sp}/{odd}"]
            fr = s["frames"][f"{sp}/{odd}"]["frames"]
            if not fr:
                continue
            rows.append([sp, odd, f"{c['total']:,}"] + [f"{c['bins'][f'{a}-{b}']:,}" for a, b in BINS] +
                        [f"{c['far']:,}", pct(c["far"] / c["total"]) if c["total"] else "n/a", f"{c['far'] / fr:.2f}",
                         pct(c["far_frames"] / fr)])
    md.append("\n## Scored cars by split, condition and distance\n\nScored cars: CAR, VAN and PICKUP labels inside the evaluation "
              "region (0-204.8 m ahead, within 25.6 m sideways) whose center projects into the front camera. Beyond 50 m means 50-200 m, as in the evaluation.\n\n" +
              table(["Split", "Condition", "Cars", "0-30 m", "30-50 m", "50-100 m", "100-200 m", "Beyond 50 m", "Share beyond 50 m",
                     "Far cars per frame", "Frames with a far car"], rows))

    rows = [[cls, f"{v['all']:,}", f"{v['scored']:,}", f"{v['scored_far']:,}",
             "scored car" if cls in splits.CAR_CLASSES else "ignored (neither positive nor negative)" if cls in splits.IGNORE_CLASSES else "not used"]
            for cls, v in sorted(s["objects"].items(), key=lambda kv: -kv[1]["all"])]
    md.append("\n## Labelled objects by class, all splits\n\n" + table(["Class", "Labels", "In evaluation region and camera view",
                                                                        "Of those, beyond 50 m", "Role in this study"], rows))

    for title, keyfn in (("LiDAR", lambda e: [pct(e["lidar_any"]), pct(e["lidar_5"]), f"{e['lidar_median']:.0f}"]),
                         ("Radar (CLOCs-R gate, 3 sweeps)", lambda e: [pct(e["radar_any"]), f"{e['radar_median_hit']:.0f}" if np.isfinite(e["radar_median_hit"]) else "n/a",
                                                                        pct(e["moving"])]),
                         ("Camera (projected 3D box)", lambda e: [f"{e['height_median']:.0f}", pct(e["small"])]),
                         ("Combined", lambda e: [pct(e["no_lidar_radar"]), pct(e["none"])])):
        header = {"LiDAR": ["Any point", "5+ points", "Median points"],
                  "Radar (CLOCs-R gate, 3 sweeps)": ["Any return", "Median returns, if any", f"Moving (Doppler > {MOVING:g} m/s), if any"],
                  "Camera (projected 3D box)": ["Median height (px)", f"Under {SMALL_PX} px"],
                  "Combined": ["Radar but no LiDAR", "Neither LiDAR nor radar"]}[title]
        rows = []
        for odd in ODDS + ("all",):
            for a, b in BINS + ((50, 200),):
                e = s["evidence"][f"{odd}/{a}-{b}"]
                if e["n"]:
                    rows.append([odd, f"{a}-{b} m" if (a, b) != (50, 200) else "**beyond 50 m**", f"{e['n']:,}"] + keyfn(e))
        md.append(f"\n## Sensor evidence on scored cars: {title}\n\n" + table(["Condition", "Distance", "Cars"] + header, rows))

    rows = [[odd, f"{v['median'] * 3.6:.0f} km/h", f"{v['lidar_points']:,.0f}", f"{v['radar_returns']:,.0f}"] for odd, v in s["ego_speed"].items()]
    md.append("\n## Per frame, median\n\n" + table(["Condition", "Ego speed", "LiDAR points", "Radar returns (3 sweeps)"], rows))
    (HERE / "dataset_tables.md").write_text("\n".join(md) + "\n")
    print("wrote", HERE / "dataset_tables.md")
    figures(o)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def figures(o):
    (HERE / "figures").mkdir(exist_ok=True)
    car = o["car"] & o["scored"]
    edges = np.arange(0, 210, 10)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), facecolor=SURFACE, sharex=True)
    for ax, odd in zip(axes, ODDS):
        d = o["dist"][car & (o["odd"] == odd)]
        ax.hist(d, edges, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.axvline(FAR, color=INK_2, linewidth=1, linestyle="--", zorder=4)
        style(ax)
        ax.set_title(f"{odd.capitalize()}: {len(d):,} cars, {100 * ((d >= FAR) & (d < 200)).mean():.0f}% at 50-200 m", color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("Distance (m)", color=INK_2, fontsize=9)
    axes[0].set_ylabel("Scored cars (all splits)", color=INK_2, fontsize=9)
    fig.suptitle("Where the labelled cars are, by driving condition (dashed line: 50 m; y-scales differ)", color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "dataset_distance.png", dpi=130)
    plt.close(fig)

    centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.7), facecolor=SURFACE, sharex=True, sharey=True)
    for ax, odd in zip(axes, ODDS):
        m = car & (o["odd"] == odd)
        idx = np.digitize(o["dist"][m], edges) - 1
        series = (("LiDAR: 1+ point in box", o["lidar"][m] > 0), ("Radar: 1+ return in gate", o["radar"][m] > 0),
                  (f"Camera: box {SMALL_PX}+ px tall", o["height_px"][m] >= SMALL_PX))
        for i, (name, ok) in enumerate(series):
            frac = [ok[idx == k].mean() if (idx == k).sum() >= 30 else np.nan for k in range(len(centers))]
            ax.plot(centers, frac, color=SERIES[i], linewidth=2, marker="o", markersize=3.5, label=name, zorder=3)
        ax.axvline(FAR, color=INK_2, linewidth=1, linestyle="--", zorder=2)
        style(ax)
        ax.set_ylim(0, 1.02)
        ax.set_title(odd.capitalize(), color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("Distance (m)", color=INK_2, fontsize=9)
    axes[0].set_ylabel("Share of scored cars", color=INK_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower center", ncol=3)
    fig.suptitle("Sensor evidence on each car, by distance (10 m bins with 30+ cars; dashed line: 50 m)", color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(HERE / "figures" / "dataset_evidence.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=SURFACE)
    for i, sp in enumerate(SPLITS):
        d = o["dist"][car & (o["split"] == sp)]
        hist, _ = np.histogram(d, edges)
        ax.plot(centers, hist / hist.sum(), color=SERIES[i], linewidth=2, marker="o", markersize=3.5, label=f"{sp} ({len(d):,} cars)", zorder=3)
    ax.axvline(FAR, color=INK_2, linewidth=1, linestyle="--", zorder=2)
    style(ax)
    ax.set_ylim(0, None)
    ax.set_xlabel("Distance (m)", color=INK_2, fontsize=9)
    ax.set_ylabel("Share of the split's scored cars", color=INK_2, fontsize=9)
    ax.set_title("Distance distribution by split (10 m bins)", color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "dataset_splits.png", dpi=130)
    plt.close(fig)
    print("wrote", HERE / "figures" / "dataset_*.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="only rebuild the tables and figures from runs/dataset_stats/")
    args = ap.parse_args()
    if not args.report:
        measure()
    with np.load(RAW / "objects.npz") as z:
        o = {k: z[k] for k in z.files}
    with open(RAW / "frames.json") as f:
        frames = json.load(f)
    report(o, frames)


if __name__ == "__main__":
    main()
