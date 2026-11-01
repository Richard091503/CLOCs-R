"""Rotated-box geometry, NMS and the distance-binned metrics of Section II-E."""
import numba
import numpy as np

from clocs_r.aimotive import DIST_BINS, box_corners_bev

CD_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)  # nuScenes center-distance thresholds (m)


@numba.njit(cache=True)
def _clip_area(p, q):
    """Intersection area of two convex quads (Sutherland-Hodgman), both counter-clockwise."""
    poly = [(float(p[i, 0]), float(p[i, 1])) for i in range(4)]
    for e in range(4):
        ax, ay = q[e, 0], q[e, 1]
        bx, by = q[(e + 1) % 4, 0], q[(e + 1) % 4, 1]
        out = []
        for i in range(len(poly)):
            cx, cy = poly[i]
            dx, dy = poly[(i + 1) % len(poly)]
            cin = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) >= 0
            din = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax) >= 0
            if cin:
                out.append((cx, cy))
            if cin != din:
                den = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
                if abs(den) > 1e-12:
                    t = ((ax - cx) * (by - ay) - (ay - cy) * (bx - ax)) / -den
                    out.append((cx + t * (dx - cx), cy + t * (dy - cy)))
        poly = out
        if len(poly) == 0:
            return 0.0
    area = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2


@numba.njit(cache=True)
def _iou_bev(a, ca, b, cb, use_3d):
    na, nb = a.shape[0], b.shape[0]
    out = np.zeros((na, nb), dtype=np.float32)
    for i in range(na):
        for j in range(nb):
            # Cheap rejection on circumscribed circles
            ra = (a[i, 3] ** 2 + a[i, 4] ** 2) ** 0.5 / 2
            rb = (b[j, 3] ** 2 + b[j, 4] ** 2) ** 0.5 / 2
            if (a[i, 0] - b[j, 0]) ** 2 + (a[i, 1] - b[j, 1]) ** 2 > (ra + rb) ** 2:
                continue
            inter = _clip_area(ca[i], cb[j])
            if inter <= 0:
                continue
            va, vb = a[i, 3] * a[i, 4], b[j, 3] * b[j, 4]
            if use_3d:
                dz = min(a[i, 2] + a[i, 5] / 2, b[j, 2] + b[j, 5] / 2) - max(a[i, 2] - a[i, 5] / 2, b[j, 2] - b[j, 5] / 2)
                if dz <= 0:
                    continue
                inter, va, vb = inter * dz, va * a[i, 5], vb * b[j, 5]
            out[i, j] = inter / (va + vb - inter)
    return out


def iou_bev(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return _iou_bev(a, box_corners_bev(a), b, box_corners_bev(b), False)


def iou_3d(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return _iou_bev(a, box_corners_bev(a), b, box_corners_bev(b), True)


def nms_bev(boxes, scores, thresh=0.1, pre_max=1000, post_max=100):
    """Greedy rotated NMS in BEV; returns kept indices in descending score order."""
    order = np.argsort(-scores)[:pre_max]
    if len(order) == 0:
        return order
    ious = iou_bev(boxes[order], boxes[order])
    keep, alive = [], np.ones(len(order), bool)
    for i in range(len(order)):
        if alive[i]:
            keep.append(i)
            if len(keep) == post_max:
                break
            alive &= ious[i] <= thresh
    return order[keep]


def iou_rect(a, b):
    """IoU matrix of axis-aligned image boxes (x0, y0, x1, y1)."""
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a, area_b = (a[:, 2:4] - a[:, :2]).prod(1), (b[:, 2:4] - b[:, :2]).prod(1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms_rect(boxes, scores, thresh=0.5, post_max=100):
    """Greedy image-plane NMS; returns kept indices in descending score order."""
    order = np.argsort(-scores)
    ious = iou_rect(boxes[order], boxes[order])
    keep, alive = [], np.ones(len(order), bool)
    for i in range(len(order)):
        if alive[i]:
            keep.append(i)
            if len(keep) == post_max:
                break
            alive &= ious[i] <= thresh
    return order[keep]


def _ap(tp, n_gt):
    """All-point interpolated AP from TP flags sorted by descending score."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    tps = np.cumsum(tp)
    recall = np.concatenate([[0.0], tps / n_gt, [1.0]])
    precision = np.concatenate([[1.0], tps / np.arange(1, len(tp) + 1), [0.0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _match(gt, det, sim, higher_better, thresh):
    """Greedy match in score order. Returns (tp flags per det, matched flags per gt)."""
    tp, used = np.zeros(len(det), bool), np.zeros(len(gt), bool)
    for d in range(len(det)):
        s = sim[:, d] if len(gt) else np.zeros(0)
        cand = np.where(~used & (s >= thresh if higher_better else s <= thresh))[0]
        if len(cand):
            g = cand[np.argmax(s[cand])] if higher_better else cand[np.argmin(s[cand])]
            used[g], tp[d] = True, True
    return tp, used


def evaluate(frames, bins=DIST_BINS, iou3d_thresholds=(0.5, 0.7), ioubev_thresholds=(0.3,), cd_thresholds=CD_THRESHOLDS):
    """frames: iterable of (gt_boxes (G, 7), det_boxes (D, 7), det_scores (D,)).

    Each GT and detection falls in one distance bin by its own BEV range, and only same-bin pairs match. A far detection
    therefore cannot claim a GT across a bin boundary, and each bin's AP sees only its own detections and GT.
    Returns {bin: {metric: value}}, with "all" added for the whole region.
    """
    names = [f"all"] + [f"{lo}-{hi}" for lo, hi in bins]
    ranges = [(0.0, float("inf"))] + list(bins)
    state = {n: {"n_gt": 0, "scores": [], "ap3d": {t: [] for t in iou3d_thresholds},
                 "apbev": {t: [] for t in ioubev_thresholds}, "cd": {t: [] for t in cd_thresholds}, "rec": 0} for n in names}
    for gt, det, score in frames:
        order = np.argsort(-score)
        det, score = det[order], score[order]
        rg, rd = np.hypot(gt[:, 0], gt[:, 1]), np.hypot(det[:, 0], det[:, 1])
        for name, (lo, hi) in zip(names, ranges):
            g, d = gt[(rg >= lo) & (rg < hi)], det[(rd >= lo) & (rd < hi)]
            st = state[name]
            st["n_gt"] += len(g)
            st["scores"].append(score[(rd >= lo) & (rd < hi)])
            i3 = iou_3d(g, d) if len(g) and len(d) else np.zeros((len(g), len(d)))
            ib = iou_bev(g, d) if len(g) and len(d) else np.zeros((len(g), len(d)))
            dc = np.hypot(g[:, None, 0] - d[None, :, 0], g[:, None, 1] - d[None, :, 1]) if len(g) and len(d) else np.zeros((len(g), len(d)))
            for t in iou3d_thresholds:
                st["ap3d"][t].append(_match(g, d, i3, True, t)[0])
            for t in ioubev_thresholds:
                tp, used = _match(g, d, ib, True, t)
                st["apbev"][t].append(tp)
                st["rec"] += used.sum()
            for t in cd_thresholds:
                st["cd"][t].append(_match(g, d, dc, False, t)[0])
    out = {}
    for name in names:
        st = state[name]
        scores = np.concatenate(st["scores"]) if st["scores"] else np.zeros(0)
        order = np.argsort(-scores)
        r = {"n_gt": int(st["n_gt"]), "n_det": int(len(scores))}
        for t in iou3d_thresholds:
            r[f"AP3D@{t}"] = _ap(np.concatenate(st["ap3d"][t])[order], st["n_gt"])
        for t in ioubev_thresholds:
            r[f"APBEV@{t}"] = _ap(np.concatenate(st["apbev"][t])[order], st["n_gt"])
        r[f"recall@BEV{ioubev_thresholds[0]}"] = st["rec"] / st["n_gt"] if st["n_gt"] else float("nan")
        cds = [_ap(np.concatenate(st["cd"][t])[order], st["n_gt"]) for t in cd_thresholds]
        r["APCD"] = float(np.mean(cds))
        for t, v in zip(cd_thresholds, cds):
            r[f"APCD@{t}"] = v
        out[name] = r
    return out


HEIGHT_BINS = {"h>=40": (40, float("inf")), "25<=h<40": (25, 40), "h<25": (0, 25)}  # pixels; 40 and 25 are KITTI's limits


def evaluate_2d(frames, bins=DIST_BINS, ap_thresholds=(0.5, 0.7), recall_thresholds=(0.3, 0.5, 0.7)):
    """Image-plane metrics of a 2D detector. frames: iterable of dicts with
    gt (G, 4), gt_range (G,), ignore (I, 4), cand (C, 4) and cand_scores (C,).

    AP uses standard NMS (IoU 0.5, at most 100 boxes), since the candidates keep near-duplicates for fusion. A 2D box has no
    distance, so AP is binned by box height, and each bin matches only its own detections and GT, as in evaluate().
    Recall by distance uses every candidate, which is what fusion receives. A candidate mostly inside an ignored box is not scored.
    """
    groups = {"all": (0, float("inf")), **HEIGHT_BINS}
    state = {n: {"n_gt": 0, "scores": [], "tp": {t: [] for t in ap_thresholds}} for n in groups}
    dist = {f"{lo}-{hi}": (lo, hi) for lo, hi in bins}
    dist["all"] = (0.0, float("inf"))
    found = {n: {"n_gt": 0, **{t: 0 for t in recall_thresholds}} for n in dist}
    n_cand = []
    for fr in frames:
        gt, cand, cand_scores = fr["gt"], fr["cand"], fr["cand_scores"]
        n_cand.append(len(cand))
        best = iou_rect(gt, cand).max(1) if len(gt) and len(cand) else np.zeros(len(gt))
        for name, (lo, hi) in dist.items():
            mask = (fr["gt_range"] >= lo) & (fr["gt_range"] < hi)
            found[name]["n_gt"] += int(mask.sum())
            for t in recall_thresholds:
                found[name][t] += int((best[mask] >= t).sum())
        keep = nms_rect(cand, cand_scores) if len(cand) else np.zeros(0, int)
        det, score = cand[keep], cand_scores[keep]
        if len(fr["ignore"]) and len(det):
            lt = np.maximum(det[:, None, :2], fr["ignore"][None, :, :2])
            rb = np.minimum(det[:, None, 2:4], fr["ignore"][None, :, 2:4])
            inside = np.clip(rb - lt, 0, None).prod(-1) / np.maximum((det[:, 2:4] - det[:, :2]).prod(1, keepdims=True), 1e-9)
            det, score = det[inside.max(1) < 0.5], score[inside.max(1) < 0.5]
        hg, hd = gt[:, 3] - gt[:, 1], det[:, 3] - det[:, 1]
        for name, (lo, hi) in groups.items():
            g, d = gt[(hg >= lo) & (hg < hi)], det[(hd >= lo) & (hd < hi)]
            st = state[name]
            st["n_gt"] += len(g)
            st["scores"].append(score[(hd >= lo) & (hd < hi)])
            sim = iou_rect(g, d) if len(g) and len(d) else np.zeros((len(g), len(d)))
            for t in ap_thresholds:
                st["tp"][t].append(_match(g, d, sim, True, t)[0])
    out = {"ap": {}, "recall": {}, "median_candidates": float(np.median(n_cand)) if n_cand else 0.0}
    for name, st in state.items():
        scores = np.concatenate(st["scores"]) if st["scores"] else np.zeros(0)
        order = np.argsort(-scores)
        out["ap"][name] = {"n_gt": st["n_gt"], "n_det": int(len(scores)),
                           **{f"AP2D@{t}": _ap(np.concatenate(st["tp"][t])[order], st["n_gt"]) for t in ap_thresholds}}
    for name, f in found.items():
        out["recall"][name] = {"n_gt": f["n_gt"],
                               **{f"recall@{t}": f[t] / f["n_gt"] if f["n_gt"] else float("nan") for t in recall_thresholds}}
    return out
