"""Radar-detector boxes as CLOCs' second sensor (results/next_round/PLAN.md, Phase 7, token RP).

CLOCs pairs each SECOND candidate with the camera's 2D boxes; RP pairs it instead with the radar detector's boxes:
every radar detection of score >= MIN_SCORE whose BEV center lies within MATCH_M of the candidate's gives one cell
[affinity, radar score], affinity = 1 - d / MATCH_M, in the places of CLOCs' [IoU, 2D score]. A candidate with no such
detection keeps the reserved cell (MISSING, MISSING), as in fusion.build_cells. No camera and no radar point features.

Token RC (PLAN.md, Phase 7 LCR addendum) keeps the camera pairs and adds the radar-detector pairs beside them, in one
network: each cell is [IoU, 2D score, affinity, radar score], with MISSING in the other sensor's two places, and
train_fusion.frame_cells lays them out as [IoU, 2D score, s3D, distance, affinity, radar score].
"""
import pathlib
import pickle

import numpy as np

from clocs_r import fusion, radar_candidates

MIN_SCORE, MATCH_M = radar_candidates.RD_MIN, radar_candidates.MATCH_M


def frame_pairs(boxes, rboxes, rscores, min_score=MIN_SCORE, match_m=MATCH_M):
    """Matched pairs of one frame: (candidate index (P,), affinity (P,), radar score (P,))."""
    keep = rscores >= min_score
    rb, rs = rboxes[keep], rscores[keep]
    if len(rb) == 0 or len(boxes) == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.float32), np.zeros(0, np.float32)
    d = np.hypot(boxes[:, None, 0] - rb[None, :, 0], boxes[:, None, 1] - rb[None, :, 1])
    j, i = np.nonzero(d < match_m)
    return j.astype(np.int32), (1.0 - d[j, i] / match_m).astype(np.float32), rs[i].astype(np.float32)


def cells(n, cand, aff, score):
    """CLOCs pair cells of a frame with n candidates: matched pairs first, then one reserved cell per unmatched candidate."""
    matched = np.zeros(n, bool)
    matched[cand] = True
    rest = np.nonzero(~matched)[0]
    column = np.concatenate([cand, rest]).astype(np.int32)
    pairs = np.concatenate([np.stack([aff, score], 1), np.full((len(rest), 2), fusion.MISSING, np.float32)]).astype(np.float16)
    return column, pairs


def build(split, cells_dir, dets, base):
    """Matched pairs of every frame of a split, cached beside the cells (one file per radar detector)."""
    tag = "_".join(pathlib.Path(dets).parts[-2:])
    path = pathlib.Path(cells_dir) / f"{split}_rp_{tag}_min{MIN_SCORE}_m{MATCH_M}.pkl"
    if not path.exists():
        out = []
        for it in base:
            with np.load(pathlib.Path(dets) / it["seq"] / f"{it['frame']:07d}.npz") as z:
                out.append(frame_pairs(it["boxes"], z["boxes"], z["scores"]))
        with open(path, "wb") as f:
            pickle.dump(dict(keys=[(it["seq"], it["frame"]) for it in base], pairs=out), f, protocol=4)
    with open(path, "rb") as f:
        rp = pickle.load(f)
    assert rp["keys"] == [(it["seq"], it["frame"]) for it in base], f"{split}: radar pairs and cells frames differ"
    return rp["pairs"]


def replace_pairs(base, rp):
    """The items with CLOCs' camera pairs replaced by radar-detector pairs (the original items are left unchanged)."""
    out = []
    for it, (cand, aff, score) in zip(base, rp):
        column, pairs = cells(len(it["scores"]), cand, aff, score)
        out.append(dict(it, column=column, pairs=pairs))
    return out


def combined_cells(item, rp_frame, top2d):
    """RC cells of one frame: the camera pairs that pass the K40 floor (as fusion.filter_pairs keeps them for LC), then the
    radar-detector pairs, then one reserved cell per candidate with neither. Rows are [IoU, 2D score, affinity, radar
    score], MISSING in the absent sensor's places."""
    n = len(item["scores"])
    col, pr = fusion.filter_pairs(item["column"], item["pairs"], n, 0.0, top2d)
    real = pr[:, 0] != fusion.MISSING
    cand, aff, score = rp_frame
    m = fusion.MISSING
    cam = np.column_stack([pr[real].astype(np.float32), np.full((int(real.sum()), 2), m, np.float32)])
    rad = np.column_stack([np.full((len(cand), 2), m, np.float32), aff, score])
    matched = np.zeros(n, bool)
    matched[col[real]] = True
    matched[cand] = True
    rest = np.nonzero(~matched)[0]
    column = np.concatenate([col[real], cand, rest]).astype(np.int32)
    pairs = np.concatenate([cam, rad, np.full((len(rest), 4), m, np.float32)]).astype(np.float16)
    return column, pairs


def add_pairs(base, rp, top2d):
    """The items with camera and radar-detector cells together (token RC); the original items are left unchanged."""
    out = []
    for it, r, t in zip(base, rp, top2d):
        column, pairs = combined_cells(it, r, t)
        out.append(dict(it, column=column, pairs=pairs))
    return out
