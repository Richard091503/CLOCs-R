"""SECOND's anchor head with range-weighted losses (see range_weight.py), for OpenPCDet."""
import torch
from pcdet.models.dense_heads import AnchorHeadSingle

from clocs_r.range_weight import END, FAR_WEIGHT, START, weight


class _Weighted(torch.nn.Module):
    """Multiplies a loss function's per-anchor weights (batch, anchors) by the head's range weights."""

    def __init__(self, loss_func, head):
        super().__init__()
        self.loss_func = loss_func
        self._head = [head]  # a list, so the head is not registered as a submodule of its own loss

    def forward(self, input, target, weights):
        return self.loss_func(input, target, weights=weights * self._head[0].range_weight)


class RangeWeightedAnchorHead(AnchorHeadSingle):
    """OpenPCDet's AnchorHeadSingle with classification, box and direction losses weighted by anchor range.

    Config: RANGE_WEIGHT: {START: 40, END: 60, FAR_WEIGHT: 2.0} under DENSE_HEAD. Inference is unchanged.
    """

    def __init__(self, model_cfg, *args, **kwargs):
        super().__init__(model_cfg, *args, **kwargs)
        rw = model_cfg.get("RANGE_WEIGHT", {})
        # Same flattening as AnchorHeadTemplate.get_box_reg_layer_loss, which matches cls_preds.view(batch, -1, classes)
        anchors = torch.cat(self.anchors, dim=-3).view(-1, self.anchors[0].shape[-1])
        w = weight(torch.hypot(anchors[:, 0], anchors[:, 1]), rw.get("START", START), rw.get("END", END),
                   rw.get("FAR_WEIGHT", FAR_WEIGHT))
        self.register_buffer("range_weight", w[None].float(), persistent=False)
        self.cls_loss_func = _Weighted(self.cls_loss_func, self)
        self.reg_loss_func = _Weighted(self.reg_loss_func, self)
        self.dir_loss_func = _Weighted(self.dir_loss_func, self)
