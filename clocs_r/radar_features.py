"""Radar evidence for 3D candidates (Section II-C): the gate of Eq. 2 and the five candidate features."""
import numpy as np

V_MAX = 40.0  # m/s; scales Doppler features to about [0, 1]
RCS_LOG_RANGE = (-1.0, 3.5)  # log10 of the RCS reported by the radar; about the 0.1st and 99.9th percentiles
FEATURES = ("r", "v", "e", "delta", "o")
# Table II of the paper. R0 is camera-LiDAR CLOCs with no radar channels.
FEATURE_SETS = {
    "R0": (), "R1": ("r",), "R2": ("v",), "R3": ("r", "v"), "R4": ("r", "v", "e"),
    "R5": ("r", "delta", "o"), "R6": ("r", "v", "delta"), "R7": ("r", "v", "e", "delta", "o"),
}


def candidate_features(boxes, pts, vel, delta0=0.5, kappa=0.02, names=FEATURES):
    """The chosen features for every candidate: (N, 5) in FEATURES order, with the others left at their missing value.

    boxes: (N, 7) [x, y, z, l, w, h, yaw] in the body frame; pts, vel: accumulated radar returns.
    With no return in the gate, r = 0 and the other features are -1, marking missing evidence.
    """
    n = len(boxes)
    out = np.full((n, 5), -1.0, dtype=np.float32)
    out[:, 0] = 0.0
    if n == 0 or len(pts) == 0:
        return out
    rho = np.hypot(boxes[:, 0], boxes[:, 1])
    delta = delta0 + kappa * rho
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    dx = pts[None, :, 0] - boxes[:, None, 0]  # (N, Q)
    dy = pts[None, :, 1] - boxes[:, None, 1]
    lx = dx * c[:, None] + dy * s[:, None]
    ly = -dx * s[:, None] + dy * c[:, None]
    gate = (np.abs(lx) <= boxes[:, None, 3] / 2 + delta[:, None]) & (np.abs(ly) <= boxes[:, None, 4] / 2 + delta[:, None])
    hit = np.nonzero(gate.any(1))[0]
    if len(hit) == 0:
        return out
    if "r" in names:
        out[hit, 0] = np.log1p(gate[hit].sum(1))
    if "delta" in names:
        q_range = np.hypot(pts[:, 0], pts[:, 1])
    if "o" in names:
        dist = np.hypot(dx, dy)
    for j in hit if set(names) - {"r"} else ():
        q = gate[j]
        if "v" in names or "e" in names:
            v = vel[q]
        if "v" in names:
            out[j, 1] = np.median(np.abs(v)) / V_MAX
        if "e" in names:
            out[j, 2] = np.median(np.abs(v - np.median(v))) / V_MAX
        if "delta" in names:
            out[j, 3] = abs(np.median(q_range[q]) - rho[j]) / delta[j]
        if "o" in names:
            out[j, 4] = dist[j][q].min() / delta[j]
    return out


def candidate_rcs(boxes, pts, rcs, delta0=0.5, kappa=0.02):
    """Median radar cross-section of each candidate's gated returns, on a log scale of about [0, 1], or -1 with no return: (N,)."""
    out = np.full(len(boxes), -1.0, dtype=np.float32)
    if len(boxes) == 0 or len(pts) == 0:
        return out
    delta = delta0 + kappa * np.hypot(boxes[:, 0], boxes[:, 1])
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    dx = pts[None, :, 0] - boxes[:, None, 0]
    dy = pts[None, :, 1] - boxes[:, None, 1]
    gate = (np.abs(dx * c[:, None] + dy * s[:, None]) <= boxes[:, None, 3] / 2 + delta[:, None]) & \
           (np.abs(-dx * s[:, None] + dy * c[:, None]) <= boxes[:, None, 4] / 2 + delta[:, None])
    for j in np.nonzero(gate.any(1))[0]:
        out[j] = (np.log10(max(np.median(rcs[gate[j]]), 10 ** RCS_LOG_RANGE[0])) - RCS_LOG_RANGE[0]) / (RCS_LOG_RANGE[1] - RCS_LOG_RANGE[0])
    return out


def select(features, feature_set):
    """Keep the columns of a Table II feature set, in FEATURES order."""
    names = FEATURE_SETS[feature_set]
    return features[:, [FEATURES.index(f) for f in names]]
