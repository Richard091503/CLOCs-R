"""Qualitative figure of the paper and its event counts, from the cached detections and the stored fusion weights.

Models: SECOND (its cached candidates), LC = R0+T2+K40 seed 1 and CLOCs-R = R0+T2+K40+RC seed 1 (docs/MODELS.md; the
code calls CLOCs-R "LCR"), from results/lcr/weights. Nothing is trained and no existing file is changed; every output is
under results/qualitative/.

    python -m clocs_r.make_qualitative            # all steps: check, thresholds, events, examples, figure
    python -m clocs_r.make_qualitative figure     # only redraw the figure from examples.json and thresholds.json
    python -m clocs_r.make_qualitative figure --bev 20 --pick radar_rescue=4   # a variant: 20 m window, alternate 4

Steps
1. Thresholds (select only): for each model, the score threshold that maximizes F1 for cars at 50-200 m, on the final
   detections (evaluation mask, BEV NMS at 0.1, as train_fusion.predict). A detection matches a car within 2 m BEV centre
   distance, greedily by score, one detection per car (metrics._match); detections and cars each fall in 50-200 m by
   their own range, as metrics.evaluate bins them.
2. Events (select and val): for every car at 50-200 m, its SECOND candidate is found among the scored candidates (the
   evaluation mask) before NMS: candidates in descending SECOND score each take the nearest unmatched car within 2 m
   (metrics._match). That candidate's SECOND, LC and CLOCs-R scores and the cell that wins CLOCs-R's max (camera, radar
   or reserved) are recorded, and each car is classified as in the paper's prompt (radar rescue, camera suppression,
   no proposal).
3. Examples (val): for each event type, among its events whose car projects at least 4 px wide, the one whose distance
   is closest to the median distance of all val events of that type; then the next four by the same rule, each at least
   MIN_GAP frames from any example already listed in the same sequence.
4. The figure: one IEEE column, three rows (a)-(c), camera crop and BEV window.
"""
import json
import pathlib
import sys

import numpy as np
import torch

from clocs_r import aimotive, cache_cells, fusion, metrics, radar_dataset, radar_pairs, splits, train_fusion, tune_fusion
from clocs_r import visibility

R = pathlib.Path("runs/lcr")
CELLS, CAND3D, CAND2D, D2 = R / "cells", R / "cand/3d", R / "cand/2d", R / "d2_dets"
WEIGHTS = pathlib.Path("results/lcr/weights")
STORED = pathlib.Path("results/lcr/data")
OUT = pathlib.Path("results/qualitative")
SEED = 1
LC, LCR = "R0+T2+K40", "R0+T2+K40+RC"
FAR_WEIGHT = 2.0  # the networks were trained with it; load_net checks it
NEAR, FAR = 50.0, 200.0
BINS = {"50-100": (50.0, 100.0), "100-200": (100.0, 200.0)}
MATCH = 2.0  # m, BEV centre distance
D2_NO_PROPOSAL = 0.1  # D2 score floor of the "no proposal" event
MARGIN = 0.5  # m, box enlargement for the LiDAR and radar counts
SWEEPS = radar_dataset.SWEEPS  # 5, with Doppler motion correction: D2's own input
MIN_PX, MIN_GAP, N_ALT = 4.0, 10, 4
EVENTS = ("radar_rescue", "camera_suppression", "no_proposal")
TITLES = {"radar_rescue": "Radar rescue", "camera_suppression": "Camera suppression", "no_proposal": "No proposal"}
CELL_NAMES = {0: "camera", 1: "radar", 2: "reserved"}


# ---------------------------------------------------------------- scoring

def load_net(config, device):
    return tune_fusion.load_net(tune_fusion.parse(config), config, WEIGHTS, SEED, FAR_WEIGHT, device)


def cell_type(pairs):
    """RC cell rows [IoU, 2D score, affinity, radar score] -> 0 camera, 1 radar, 2 reserved."""
    m = fusion.MISSING
    return np.where(pairs[:, 0] != m, 0, np.where(pairs[:, 2] != m, 1, 2))


@torch.no_grad()
def score_split(split, device):
    """Per frame: the scored candidates with their three scores and CLOCs-R's winning cell, the cars, and each model's
    final detections (evaluation mask, BEV NMS at 0.1)."""
    base, _ = cache_cells.load(split, CELLS, *tune_fusion.GATE, cache_cells.SWEEPS, use_radar=False)
    top2d = visibility.load(split, CELLS, base)["top2d"]
    rp = radar_pairs.build(split, CELLS, D2, base)  # the cached pairs; nothing is rebuilt
    lc, lcr = load_net(LC, device), load_net(LCR, device)
    lc.eval(), lcr.eval()
    frames = []
    for i, it in enumerate(base):
        n = len(it["scores"])
        x, col = train_fusion.frame_cells(it, device, (0.0, top2d[i]))
        s_lc = torch.sigmoid(lc(x, col, n)).cpu().numpy()
        rc_col, rc_pairs = radar_pairs.combined_cells(it, rp[i], top2d[i])
        x, col = train_fusion.frame_cells(dict(it, column=rc_col, pairs=rc_pairs), device)
        z = torch.logit(x[:, 2].clamp(1e-6, 1 - 1e-6))
        cell = (lcr.net(x.t()[None, :, None, :])[0, 0, 0] + z).cpu().numpy()  # FusionNet.forward before its max
        c = col.cpu().numpy()
        order = np.lexsort((cell, c))
        sc = c[order]
        last = np.r_[sc[1:] != sc[:-1], True]
        win = np.full(n, -1, np.int64)
        win[sc[last]] = order[last]
        logit = np.full(n, -1e6, np.float32)
        logit[sc[last]] = cell[order[last]]
        ref = lcr(x, col, n).cpu().numpy()
        assert np.allclose(logit, ref, atol=1e-5), f"{split} frame {i}: per-cell max differs from FusionNet"
        s_lcr = 1 / (1 + np.exp(-logit.astype(np.float64)))
        wp = rc_pairs[win].astype(np.float32)
        m = it["keep_eval"]
        f = dict(seq=it["seq"], frame=int(it["frame"]), odd=it["odd"], gt=it["gt"], boxes=it["boxes"][m],
                 s={"SECOND": it["scores"][m], "LC": s_lc[m], "CLOCs-R": s_lcr[m].astype(np.float32)},
                 win=cell_type(wp)[m], win_pairs=wp[m])
        f["final"] = {}
        for name, s in f["s"].items():
            keep = metrics.nms_bev(f["boxes"], s, 0.1)
            f["final"][name] = (f["boxes"][keep], s[keep])
        frames.append(f)
    del base, rp
    return frames


def reproduction(frames, split):
    """APCD of the three models from this script against the stored results of the same seed (no ranking change allowed)."""
    stored = {"SECOND": json.load(open(STORED / "aimotive_second.json"))[split]["overall"],
              "LC": _seed(json.load(open(STORED / "final_r0_s123.json"))[LC], split),
              "CLOCs-R": _seed(json.load(open(STORED / ("lcr.json" if split == "select" else "lcr_val.json")))[LCR], split)}
    out = {}
    for name in ("SECOND", "LC", "CLOCs-R"):
        r = metrics.evaluate([(f["gt"], *f["final"][name]) for f in frames], bins=aimotive.DIST_BINS + [train_fusion.FAR])
        out[name] = {b: dict(here=round(r[b]["APCD"], 6), stored=round(stored[name][b]["APCD"], 6),
                             diff=round(r[b]["APCD"] - stored[name][b]["APCD"], 6)) for b in ("all", "50-100", "100-200", "50-200")}
    return out


def _seed(entry, split):
    return [r for r in entry["runs"] if r["seed"] == SEED][0][split]["overall"]


# ---------------------------------------------------------------- step 1: thresholds

def far(boxes):
    r = np.hypot(boxes[:, 0], boxes[:, 1])
    return (r >= NEAR) & (r < FAR)


def f1_curve(frames, name):
    """(scores descending, tp flags, n_gt) of the final far detections, greedy-matched within 2 m per frame."""
    scores, tps, n_gt = [], [], 0
    for f in frames:
        b, s = f["final"][name]
        k = far(b)
        b, s = b[k], s[k]
        o = np.argsort(-s)
        b, s = b[o], s[o]
        g = f["gt"][far(f["gt"])] if len(f["gt"]) else f["gt"]
        n_gt += len(g)
        d = np.hypot(g[:, None, 0] - b[None, :, 0], g[:, None, 1] - b[None, :, 1]) if len(g) and len(b) else np.zeros((len(g), len(b)))
        tp, _ = metrics._match(g, b, d, False, MATCH)
        scores.append(s)
        tps.append(tp)
    s, tp = np.concatenate(scores), np.concatenate(tps)
    o = np.argsort(-s, kind="stable")
    return s[o], tp[o], n_gt


def at_threshold(s, tp, n_gt, t):
    """Greedy matching in score order means that keeping only detections >= t keeps exactly their matches."""
    k = s >= t
    n_tp, n_det = int(tp[k].sum()), int(k.sum())
    p, r = n_tp / max(n_det, 1), n_tp / max(n_gt, 1)
    return dict(threshold=float(t), F1=2 * n_tp / max(n_det + n_gt, 1), precision=p, recall=r, tp=n_tp, detections=n_det, cars=n_gt)


def best_threshold(frames, name):
    s, tp, n_gt = f1_curve(frames, name)
    ctp = np.cumsum(tp)
    last = np.r_[s[1:] != s[:-1], True]  # a threshold keeps every detection tied with it
    f1 = 2 * ctp / (np.arange(1, len(s) + 1) + n_gt)
    i = np.nonzero(last)[0][np.argmax(f1[last])]
    return at_threshold(s, tp, n_gt, s[i])


# ---------------------------------------------------------------- step 2: events

def frame_cars(f, d2):
    """One row per car at 50-200 m: its matched SECOND candidate (index or -1) and the best D2 box within 2 m."""
    gt = f["gt"]
    if len(gt) == 0:
        return []
    b, s = f["boxes"], f["s"]["SECOND"]
    o = np.argsort(-s, kind="stable")
    d = np.hypot(gt[:, None, 0] - b[None, o, 0], gt[:, None, 1] - b[None, o, 1]) if len(b) else np.zeros((len(gt), 0))
    used = np.zeros(len(gt), bool)
    match = np.full(len(gt), -1)
    for k in range(len(o)):  # metrics._match, keeping which candidate claimed each car
        cand = np.nonzero(~used & (d[:, k] <= MATCH))[0]
        if len(cand):
            g = cand[np.argmin(d[cand, k])]
            used[g], match[g] = True, o[k]
    rb, rs = d2
    rows = []
    for g in np.nonzero(far(gt))[0]:
        dd = np.hypot(rb[:, 0] - gt[g, 0], rb[:, 1] - gt[g, 1]) if len(rb) else np.zeros(0)
        near = np.nonzero(dd <= MATCH)[0]
        best = near[np.argmax(rs[near])] if len(near) else -1
        rows.append(dict(g=int(g), cand=int(match[g]), d2=int(best),
                         any_candidate=bool(len(b) and (np.hypot(b[:, 0] - gt[g, 0], b[:, 1] - gt[g, 1]) <= MATCH).any())))
    return rows


def detected(final, car, t):
    b, s = final
    k = s >= t
    return bool(k.any() and (np.hypot(b[k, 0] - car[0], b[k, 1] - car[1]) <= MATCH).any())


def counts_in_box(seq, frame, car):
    """LiDAR points inside the car's 3D box enlarged by MARGIN on every side; radar returns of the last SWEEPS sweeps (D2's
    input, motion-corrected) inside its BEV box enlarged by MARGIN on every side (the clutter label of radar_clutter)."""
    big = car.copy()
    big[3:6] += 2 * MARGIN
    n_lidar = int(visibility.points_in_boxes(big[None], seq.lidar(frame))[0])
    pts = radar_dataset.radar_points(seq, frame, SWEEPS, motion=True)
    dx, dy = pts[:, 0] - car[0], pts[:, 1] - car[1]
    c, s = np.cos(car[6]), np.sin(car[6])
    inside = (np.abs(dx * c + dy * s) <= car[3] / 2 + MARGIN) & (np.abs(-dx * s + dy * c) <= car[4] / 2 + MARGIN)
    return n_lidar, int(inside.sum())


def width_px(seq, car):
    rect, valid = aimotive.project_boxes(seq, car[None])
    return float(rect[0, 2] - rect[0, 0]) if valid[0] else 0.0


def events(frames, split, thr):
    seqs = {pathlib.Path(p).name: p for p in splits.scenes(split)}
    seq_obj = {}
    records, shares = [], {b: np.zeros(3, int) for b in BINS}
    after_nms = {e: dict(n=0, second_detected=0, clocsr_detected=0) for e in EVENTS}
    for f in frames:
        with np.load(D2 / f["seq"] / f"{f['frame']:07d}.npz") as z:
            d2 = (z["boxes"], z["scores"])
        for row in frame_cars(f, d2):
            car = f["gt"][row["g"]]
            dist = float(np.hypot(car[0], car[1]))
            j = row["cand"]
            s = {k: (float(v[j]) if j >= 0 else None) for k, v in f["s"].items()}
            win = CELL_NAMES[int(f["win"][j])] if j >= 0 else None
            if j >= 0:
                shares["50-100" if dist < 100 else "100-200"][int(f["win"][j])] += 1
            kind = None
            if j >= 0 and s["SECOND"] < thr["SECOND"] and s["CLOCs-R"] >= thr["CLOCs-R"] and win == "radar":
                kind = "radar_rescue"
            elif j >= 0 and s["SECOND"] >= thr["SECOND"] and s["CLOCs-R"] < thr["CLOCs-R"] and win in ("camera", "reserved"):
                kind = "camera_suppression"
            elif not row["any_candidate"] and row["d2"] >= 0 and d2[1][row["d2"]] >= D2_NO_PROPOSAL:
                kind = "no_proposal"
            if kind is None:
                continue
            if f["seq"] not in seq_obj:
                seq_obj.clear()
                seq_obj[f["seq"]] = aimotive.Sequence(seqs[f["seq"]])
            seq = seq_obj[f["seq"]]
            n_lidar, n_radar = counts_in_box(seq, f["frame"], car)
            wp = f["win_pairs"][j] if j >= 0 else None
            rec = dict(event=kind, split=split, sequence=f["seq"], frame=f["frame"], condition=f["odd"],
                       distance_m=round(dist, 2), scores=s, winning_cell=win,
                       winning_cell_values=(None if wp is None else
                                            dict(iou=float(wp[0]), score_2d=float(wp[1])) if win == "camera" else
                                            dict(affinity=float(wp[2]), radar_score=float(wp[3])) if win == "radar" else {}),
                       lidar_points=n_lidar, radar_returns=n_radar, width_px=round(width_px(seq, car), 2),
                       d2_score=float(d2[1][row["d2"]]) if row["d2"] >= 0 else None,
                       gt_box=[round(float(v), 4) for v in car],
                       second_box=[round(float(v), 4) for v in f["boxes"][j]] if j >= 0 else None,
                       d2_box=[round(float(v), 4) for v in d2[0][row["d2"]]] if row["d2"] >= 0 else None,
                       detected_after_nms={"SECOND": detected(f["final"]["SECOND"], car, thr["SECOND"]),
                                           "CLOCs-R": detected(f["final"]["CLOCs-R"], car, thr["CLOCs-R"])})
            records.append(rec)
            a = after_nms[kind]
            a["n"] += 1
            a["second_detected"] += rec["detected_after_nms"]["SECOND"]
            a["clocsr_detected"] += rec["detected_after_nms"]["CLOCs-R"]
    cars = sum(int(far(f["gt"]).sum()) for f in frames if len(f["gt"]))
    counts = {e: sum(r["event"] == e for r in records) for e in EVENTS}
    supp = [r for r in records if r["event"] == "camera_suppression"]
    counts["camera_suppression_by_cell"] = {c: sum(r["winning_cell"] == c for r in supp) for c in ("camera", "reserved")}
    share = {b: dict(candidates=int(v.sum()), **{CELL_NAMES[k]: (float(v[k] / v.sum()) if v.sum() else None) for k in range(3)})
             for b, v in shares.items()}
    tot = sum(shares.values())
    share["50-200"] = dict(candidates=int(tot.sum()), **{CELL_NAMES[k]: float(tot[k] / tot.sum()) for k in range(3)})
    return dict(cars_50_200=cars, cars_with_candidate=int(tot.sum()), counts=counts, winning_cell_share=share,
                after_nms_check=after_nms), records


# ---------------------------------------------------------------- step 3: examples

def examples(records):
    out = dict(rule=(f"val only. Per event type, the median distance of all its val events; among events whose car's "
                     f"projected 2D box is at least {MIN_PX:g} px wide, the one closest to that median (ties: sequence, "
                     f"frame). Alternates: the next {N_ALT} by the same distance rule, skipping events within {MIN_GAP} "
                     f"frames of an already listed example in the same sequence."))
    for e in EVENTS:
        rows = [r for r in records if r["split"] == "val" and r["event"] == e]
        if not rows:
            out[e] = dict(median_distance_m=None, chosen=None, alternates=[], eligible=0, events=0)
            continue
        med = float(np.median([r["distance_m"] for r in rows]))
        ok = sorted([r for r in rows if r["width_px"] >= MIN_PX],
                    key=lambda r: (abs(r["distance_m"] - med), r["sequence"], r["frame"]))
        listed = []
        for r in ok:
            if all(r["sequence"] != q["sequence"] or abs(r["frame"] - q["frame"]) >= MIN_GAP for q in listed):
                listed.append(r)
            if len(listed) == 1 + N_ALT:
                break
        out[e] = dict(events=len(rows), eligible=len(ok), median_distance_m=med,
                      chosen=listed[0] if listed else None, alternates=listed[1:])
    return out


# ---------------------------------------------------------------- step 4: figure

W_IN, H_IN = 3.5, 3.38
ROW_H, TITLE_H, PANEL_H = 0.99, 0.13, 0.68
CAM_X, CAM_W, BEV_X = 0.03, 2.02, 2.25
BEV_M = 40.0  # m, side of the BEV window (the prompt's); --bev changes it for a variant figure
NMS_2D = 0.5  # Cascade's exported candidates kept near-duplicates (NMS 0.95, for CLOCs); drawn after the usual 2D NMS
FS = 6.5  # pt, every text in the figure


def _stroke(lw, color):
    from matplotlib import patheffects
    return [patheffects.Stroke(linewidth=lw, foreground=color), patheffects.Normal()]


def _rect_xy(x0, y0, x1, y1):
    return [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]


def camera_panel(ax, seq, rec):
    from PIL import Image
    car = np.array(rec["gt_box"], np.float32)
    rect, _ = aimotive.project_boxes(seq, car[None])
    x0, y0, x1, y1 = rect[0]
    W, H = seq.meta["image_size"]
    aspect = CAM_W / PANEL_H
    cw = max(6 * (x1 - x0), 120.0)
    ch = cw / aspect
    if ch > H:
        ch, cw = H, H * aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = float(np.clip(cx - cw / 2, 0, W - cw))
    top = float(np.clip(cy - ch / 2, 0, H - ch))
    img = Image.open(seq.image_path(rec["frame"])).convert("RGB")  # pixels only: no EXIF or other metadata is carried
    crop = img.crop((round(left), round(top), round(left + cw), round(top + ch)))
    crop = crop.resize((int(CAM_W * 300), int(PANEL_H * 300)), Image.BICUBIC)
    ax.imshow(np.asarray(crop), extent=(left, left + cw, top + ch, top), interpolation="none")
    ax.set_xlim(left, left + cw)
    ax.set_ylim(top + ch, top)
    det = np.load(CAND2D / rec["sequence"] / f"{rec['frame']:07d}.npy")
    det = det[metrics.nms_rect(det[:, :4], det[:, 4], NMS_2D)]
    det = det[det[:, 4] > 0.05]
    det = det[(det[:, 2] > left) & (det[:, 0] < left + cw) & (det[:, 3] > top) & (det[:, 1] < top + ch)]
    for b in det:
        xs, ys = _rect_xy(*b[:4])
        ax.plot(xs, ys, color="white", lw=0.5, ls=(0, (2.5, 1.5)), path_effects=_stroke(1.2, "black"))
    # Score labels, highest score first; a label goes above its box, or below it if that spot is taken or off the crop.
    lw_, lh_ = 0.11 * cw, 0.17 * ch  # label size in data units at FS (four characters, with padding)
    taken = []

    def clear_of_gt(x, y0):  # the label must not cover the ground-truth box's outline
        return x + lw_ < x0 - 2 or x > x1 + 2 or y0 + lh_ < y0_gt - 2 or y0 > y1_gt + 2 or \
            (x > x0 + 2 and x + lw_ < x1 - 2 and y0 > y0_gt + 2 and y0 + lh_ < y1_gt - 2)

    y0_gt, y1_gt = y0, y1
    for b in det[np.argsort(-det[:, 4])]:
        spots = [(b[0], b[1], "bottom"), (b[0], b[3], "top"), (b[2] - lw_, b[1], "bottom"), (b[2] - lw_, b[3], "top"),
                 (b[0] - lw_, b[1] + lh_, "bottom"), (b[2], b[1] + lh_, "bottom")]
        for x, y, va in spots:
            y0l = y - lh_ if va == "bottom" else y
            free = all(abs(x - tx) > lw_ or abs(y0l - ty) > lh_ for tx, ty in taken) and clear_of_gt(x, y0l)
            if free and y0l >= top and y0l + lh_ <= top + ch and left <= x <= left + cw - lw_:
                taken.append((x, y0l))
                ax.text(x, y, f"{b[4]:.2f}", fontsize=FS, color="white", ha="left", va=va,
                        bbox=dict(boxstyle="square,pad=0.08", fc="black", ec="none"))
                break
    xs, ys = _rect_xy(x0, y0, x1, y1)
    ax.plot(xs, ys, color="black", lw=1.4, path_effects=_stroke(2.6, "white"), solid_joinstyle="miter", zorder=6)
    ax.set_xticks([]), ax.set_yticks([])
    return det


def bev_panel(ax, seq, rec, side):
    from matplotlib.patches import Polygon
    car = np.array(rec["gt_box"], np.float32)
    h = side / 2

    def to_xy(p):  # body x forward -> up, body y left -> left
        return np.stack([-(p[..., 1] - car[1]), p[..., 0] - car[0]], -1)

    q = to_xy(seq.lidar(rec["frame"])[:, :3])
    q = q[(np.abs(q) <= h).all(1)]
    ax.scatter(q[:, 0], q[:, 1], s=0.35, c="0.45", marker="o", linewidths=0, rasterized=True)
    r = to_xy(radar_dataset.radar_points(seq, rec["frame"], SWEEPS, motion=True)[:, :3])
    r = r[(np.abs(r) <= h).all(1)]
    ax.scatter(r[:, 0], r[:, 1], s=5, c="black", marker="x", linewidths=0.5)
    styles = [("gt_box", dict(color="black", lw=1.0)),
              ("second_box", dict(color="black", lw=0.7, ls=(0, (1.8, 1.0)), path_effects=_stroke(1.5, "white"))),
              ("d2_box", dict(color="black", lw=0.8, ls=(0, (0.5, 0.8)), path_effects=_stroke(1.6, "white")))]
    for key, kw in styles:
        if rec.get(key) is None:
            continue
        c = to_xy(aimotive.box_corners_bev(np.array(rec[key], np.float32)[None])[0])
        ax.add_patch(Polygon(c, closed=True, fill=False, joinstyle="miter", **kw))
    bar = 10.0 if side >= 30 else 5.0
    ax.plot([-h + 0.05 * side, -h + 0.05 * side + bar], [-h + 0.055 * side] * 2, color="black", lw=1.0)
    ax.text(-h + 0.05 * side + bar / 2, -h + 0.08 * side, f"{bar:.0f} m", fontsize=FS, ha="center", va="bottom",
            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    ax.set_xlim(-h, h), ax.set_ylim(-h, h)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])


def score_text(rec):
    s = rec["scores"]
    if rec["event"] == "no_proposal":
        return f"No SECOND candidate\nD2 (radar) {rec['d2_score']:.2f}"
    # LC is shown as well: CLOCs-R's threshold differs from SECOND's, and the radar or camera cell's effect is clearest
    # against the camera-only network (in most radar rescues CLOCs-R is below SECOND but above LC).
    return f"SECOND {s['SECOND']:.2f}, LC {s['LC']:.2f}\n→ CLOCs-R {s['CLOCs-R']:.2f} ({rec['winning_cell']} cell)"


def figure(ex, thresholds, side=BEV_M, pick=None):
    """pick: {event: index}, 0 = the rule's choice, 1-4 = its alternates in examples.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                                "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "font.size": FS,
                                "axes.linewidth": 0.5})
    pick = pick or {}
    scenes = {pathlib.Path(p).name: p for p in splits.scenes("val")}
    fig = plt.figure(figsize=(W_IN, H_IN))
    used = []
    for k, e in enumerate(EVENTS):
        options = [ex[e]["chosen"]] + ex[e]["alternates"]
        rec = options[pick.get(e, 0)]
        if rec is None:
            raise SystemExit(f"no eligible {e} event on val")
        seq = aimotive.Sequence(scenes[rec["sequence"]])
        top = H_IN - k * ROW_H
        fig.text(CAM_X / W_IN, (top - 0.01) / H_IN,
                 f"({'abc'[k]}) {TITLES[e]}, {rec['distance_m']:.0f} m, {rec['condition']}",
                 fontsize=FS + 0.5, fontweight="bold", va="top", ha="left")
        y = (top - TITLE_H - PANEL_H) / H_IN
        cam = fig.add_axes([CAM_X / W_IN, y, CAM_W / W_IN, PANEL_H / H_IN])
        det = camera_panel(cam, seq, rec)
        bev = fig.add_axes([BEV_X / W_IN, y, PANEL_H / W_IN, PANEL_H / H_IN])
        bev_panel(bev, seq, rec, side)
        fig.text((BEV_X + PANEL_H / 2) / W_IN, y - 0.02 / H_IN, score_text(rec), fontsize=FS, ha="center", va="top",
                 linespacing=1.1)
        used.append(dict(event=e, index=pick.get(e, 0), sequence=rec["sequence"], frame=rec["frame"],
                         cascade_boxes_drawn=int(len(det))))
    handles = [Line2D([], [], color="black", lw=1.0, label="Ground truth"),
               Line2D([], [], color="black", lw=0.7, ls=(0, (1.8, 1.0)), label="SECOND"),
               Line2D([], [], color="black", lw=0.8, ls=(0, (0.5, 0.8)), label="D2 (radar)"),
               Line2D([], [], color="white", lw=0.5, ls=(0, (2.5, 1.5)), path_effects=_stroke(1.2, "black"),
                      label="Cascade R-CNN"),
               Line2D([], [], color="0.45", marker="o", ls="none", ms=1.2, label="LiDAR"),
               Line2D([], [], color="black", marker="x", ls="none", ms=3, mew=0.6, label="Radar, 5 sweeps")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.1 / H_IN), ncol=3, fontsize=FS,
               frameon=False, handlelength=2.2, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0,
               labelspacing=0.2)
    fig.text(0.5, 0.005, f"Thresholds (max F1, 50–200 m, select): SECOND {thresholds['SECOND']:.2f}, "
                         f"LC {thresholds['LC']:.2f}, CLOCs-R {thresholds['CLOCs-R']:.2f}", fontsize=FS, ha="center",
             va="bottom")
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "qualitative" if side == BEV_M else f"qualitative_bev{side:g}"
    meta = {"Creator": None, "Producer": None, "Author": None, "Title": None, "CreationDate": None}
    fig.savefig(OUT / f"{stem}.pdf", dpi=300, metadata=meta)  # dpi: the rasterized LiDAR dots
    fig.savefig(OUT / f"{stem}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    dump(dict(bev_window_m=side, picks=used, nms_2d_for_drawing=NMS_2D), f"{stem}_figure.json")
    return used


# ---------------------------------------------------------------- main

def dump(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=1)


def main():
    if sys.argv[1:2] == ["figure"]:  # figure [--bev 20] [--pick radar_rescue=4 ...]
        args = sys.argv[2:]
        side = float(args[args.index("--bev") + 1]) if "--bev" in args else BEV_M
        pick = {a.split("=")[0]: int(a.split("=")[1]) for a in args if "=" in a}
        t = json.load(open(OUT / "thresholds.json"))["thresholds"]
        print(figure(json.load(open(OUT / "examples.json")), t, side, pick))
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sel = score_split("select", device)
    check = {"select": reproduction(sel, "select")}
    thr = {name: best_threshold(sel, name) for name in ("SECOND", "LC", "CLOCs-R")}
    t = {k: v["threshold"] for k, v in thr.items()}
    summary, records = {}, []
    summary["select"], rec = events(sel, "select", t)
    records += rec
    del sel
    val = score_split("val", device)
    check["val"] = reproduction(val, "val")
    val_f1 = {}
    for name in t:
        s, tp, n_gt = f1_curve(val, name)
        val_f1[name] = at_threshold(s, tp, n_gt, t[name])
    summary["val"], rec = events(val, "val", t)
    records += rec
    del val
    dump(check, "reproduction.json")
    dump(dict(rule=("F1-maximizing score threshold for cars at 50-200 m on select; final detections (evaluation mask, "
                    "BEV NMS 0.1); BEV centre distance <= 2 m, greedy by score, one detection per car; detections and "
                    "cars binned by their own range. Applied unchanged to val."),
              seed=SEED, thresholds=t, select=thr, val_at_select_threshold=val_f1), "thresholds.json")
    dump(dict(thresholds=t, definitions=dict(
        matching=("per car at 50-200 m, the scored SECOND candidates (evaluation mask) before NMS, in descending SECOND "
                  "score, each take the nearest unmatched car within 2 m (metrics._match)"),
        radar_rescue="SECOND score < SECOND threshold, CLOCs-R score >= CLOCs-R threshold, winning cell is a radar cell",
        camera_suppression=("SECOND score >= SECOND threshold, CLOCs-R score < CLOCs-R threshold, winning cell is a "
                            "camera cell or the reserved cell"),
        no_proposal="no scored SECOND candidate within 2 m, and a D2 box of score >= 0.1 within 2 m",
        lidar_points="inside the car's 3D box enlarged by 0.5 m on every side",
        radar_returns="last 5 sweeps with Doppler motion correction (D2's input), inside the BEV box enlarged by 0.5 m",
        width_px="width of the car's projected 3D box, clipped to the image; 0 if not projectable",
        after_nms_check=("not part of the event rule: whether a final detection (after NMS) of score >= threshold lies "
                         "within 2 m of the car, for SECOND and CLOCs-R")),
        summary=summary, records=records), "events.json")
    ex = examples(records)
    dump(ex, "examples.json")
    drawn = figure(ex, t)
    print(json.dumps(dict(thresholds=t, summary=summary, drawn=drawn), indent=1))


if __name__ == "__main__":
    main()
