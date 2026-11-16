"""Range-weighted losses: training emphasis on objects beyond 50 m, shared by SECOND and the fusion layer.

w(r) = 1 below START, FAR_WEIGHT beyond END, linear in between, where r is the BEV distance from the ego car. The ramp
(40-60 m by default) centres the switch on 50 m without a step in the loss at the bin boundary. Every anchor or candidate
is weighted by its own range, positives and negatives alike, so the far region counts more as a whole (recall and
precision); the loss is still normalized by the number of positives, as in SECOND and CLOCs.

This module has no OpenPCDet import, so the fusion code can use it in forked CUDA workers; SECOND's head is in second_head.py.
"""
import numpy as np
import torch

START, END, FAR_WEIGHT = 40.0, 60.0, 2.0


def weight(r, start=START, end=END, far_weight=FAR_WEIGHT):
    """Per-element loss weight for BEV ranges r (numpy array or torch tensor)."""
    lib = torch if isinstance(r, torch.Tensor) else np
    ramp = lib.clip((r - start) / (end - start), 0.0, 1.0)
    return 1.0 + (far_weight - 1.0) * ramp
