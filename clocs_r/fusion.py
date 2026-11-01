"""CLOCs-R fusion network and pair-tensor construction (Section II-D, Eqs. 3-4)."""
import numpy as np
import torch
from torch import nn

MISSING = -1.0  # IoU and 2D score of the reserved no-camera row, as stated in the CLOCs paper


class FusionNet(nn.Module):
    """Four 1x1 convolutions, (4+m) -> 18 -> 36 -> 36 -> 1, the same layers as the CLOCs fuse_2d_3d block."""

    def __init__(self, num_radar=0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4 + num_radar, 18, 1), nn.ReLU(), nn.Conv2d(18, 36, 1), nn.ReLU(),
                                 nn.Conv2d(36, 36, 1), nn.ReLU(), nn.Conv2d(36, 1, 1))

    def forward(self, cells, column, n):
        """cells: (P, 4+m) non-empty cells; column: (P,) candidate index of each cell; n: candidates in the frame.

        Returns (n,) logits: the max over each candidate's cells (Eq. 4).
        """
        x = self.net(cells.t()[None, :, None, :])[0, 0, 0]
        return _max_by_column(x, column, n)


def _max_by_column(x, column, n):
    # torch 1.10 has no scatter_reduce, and repeated indices in a write have no defined order on any device. Sort by value,
    # then stably by column, so the last cell of each column is its maximum, and write only those.
    by_value = torch.argsort(x)
    by_column = torch.sort(column[by_value], stable=True).indices
    order = by_value[by_column]
    sorted_column = column[order]
    last = torch.ones_like(sorted_column, dtype=torch.bool)
    last[:-1] = sorted_column[1:] != sorted_column[:-1]
    out = x.new_full((n,), -1e6)
    out[sorted_column[last]] = x[order[last]]
    return out


def iou_2d_matrix(a, b):
    """Axis-aligned IoU, (K, 4) x (N, 4) -> (K, N)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    iy = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-6)


def build_cells(boxes2d, scores2d, proj2d, scores3d, dist, radar):
    """Non-empty cells of the k x n tensor (Eq. 3).

    boxes2d, scores2d: k 2D candidates (none when the camera is unavailable).
    proj2d: (n, 4) image-plane boxes of the n 3D candidates; NaN when a candidate is not projectable.
    scores3d, dist: per-candidate 3D score and normalized BEV distance; radar: (n, m) radar channels.
    Every candidate with no overlapping 2D box gets one cell in the reserved row (IoU = s2D = MISSING).
    Returns cells (P, 4+m) and the candidate index of each cell (P,).
    """
    n, m = len(scores3d), radar.shape[1]
    iou = iou_2d_matrix(boxes2d, np.nan_to_num(proj2d))
    i, j = np.nonzero(iou > 0)
    matched = np.zeros(n, bool)
    matched[j] = True
    rest = np.nonzero(~matched)[0]
    column = np.concatenate([j, rest])
    cells = np.empty((len(column), 4 + m), dtype=np.float32)
    cells[:, 0] = np.concatenate([iou[i, j], np.full(len(rest), MISSING)])
    cells[:, 1] = np.concatenate([scores2d[i], np.full(len(rest), MISSING)]) if len(scores2d) else MISSING
    cells[:, 2] = scores3d[column]
    cells[:, 3] = dist[column]
    if m:
        cells[:, 4:] = radar[column]
    return cells, column


def focal_loss(logits, targets, weights, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss, summed; targets 1/0, weight 0 for the ignored IoU band."""
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    a = alpha * targets + (1 - alpha) * (1 - targets)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (a * (1 - pt) ** gamma * ce * weights).sum()
