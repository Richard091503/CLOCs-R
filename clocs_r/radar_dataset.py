"""aiMotive radar as an OpenPCDet dataset, for a radar-only PointPillars (results/next_round/PLAN.md, Phase 6).

Per point: x, y, z in the current frame's body coordinates, Doppler speed with ego motion removed (Sequence.radar_sweep)
over V_SCALE, log RCS scaled as radar_features.RCS_LOG_RANGE, and sweep age in seconds; SWEEPS sweeps accumulated with
the ego poses (no Doppler motion correction), View-of-Delft style.
"""
import numpy as np
import torch
from pcdet.datasets import DatasetTemplate
from pcdet.models.backbones_2d.map_to_bev import PointPillarScatter

from clocs_r import aimotive, radar_features, splits

SWEEPS = 5
V_SCALE = radar_features.V_MAX  # 40 m/s
GATE = (0.5, 0.02)  # a training label needs at least one return in this gate (radar_features.candidate_features)
FEATURES = ("x", "y", "z", "v", "rcs", "age")


def radar_points(seq, frame, m=SWEEPS, motion=False):
    """(N, 6) float32 radar points of the last m sweeps in FEATURES order (the same accumulation as Sequence._accumulate).

    motion (Phase 6b, D2): each older return is first moved along its line of sight by its Doppler speed times its age,
    as Sequence._accumulate(motion=True) does."""
    poses, rad = seq.meta["poses"], seq.radar
    lo, hi = radar_features.RCS_LOG_RANGE
    out = []
    for k, f in enumerate(range(frame, frame - m, -1)):
        p, v = seq.radar_sweep(f)
        if len(p) == 0:
            continue
        if f != frame:
            if f not in poses or frame not in poses:
                continue
            if motion:
                p = p.copy()
                sight = p[:, :2] / np.maximum(np.linalg.norm(p[:, :2], axis=1, keepdims=True), 1e-3)
                p[:, :2] += sight * (v * (frame - f) * seq.frame_period)[:, None]
            T = np.linalg.inv(poses[frame]) @ poses[f]
            p = p @ T[:3, :3].T + T[:3, 3]
        i = rad["index"][f]
        rcs = rad["rcs"][rad["offsets"][i]:rad["offsets"][i + 1]]
        r = (np.log10(np.maximum(rcs, 10 ** lo)) - lo) / (hi - lo)
        out.append(np.column_stack([p, v / V_SCALE, r, np.full(len(p), k * seq.frame_period)]))
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, len(FEATURES)), np.float32)


def radar_labels(seq, frame, points):
    """Car boxes in the camera view with at least one radar return in the gate: the radar detector's training labels."""
    boxes, _ = seq.boxes(frame, splits.CAR_CLASSES)
    if len(boxes):
        boxes = boxes[aimotive.in_camera_view(seq, boxes)]
    if len(boxes):
        hits = radar_features.candidate_features(boxes, points[:, :3], points[:, 3] * V_SCALE, *GATE, names=("r",))[:, 0]
        boxes = boxes[hits > 0]
    return boxes


class RadarPillarScatter(PointPillarScatter):
    """OpenPCDet's PointPillarScatter, with the batch size taken from the batch instead of the largest frame index that
    has a pillar: radar frames can have no point in range, and a batch ending in such frames came out too small (export
    crashed on 2026-11-20). Identical output for every other batch; no parameters. train_second.load_cfg registers it
    under the name PointPillarScatter."""

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict["pillar_features"], batch_dict["voxel_coords"]
        out = []
        for b in range(batch_dict["batch_size"]):
            spatial = pillar_features.new_zeros(self.num_bev_features, self.nz * self.nx * self.ny)
            mask = coords[:, 0] == b
            c = coords[mask]
            spatial[:, (c[:, 1] + c[:, 2] * self.nx + c[:, 3]).long()] = pillar_features[mask].t()
            out.append(spatial)
        batch_dict["spatial_features"] = torch.stack(out, 0).view(len(out), self.num_bev_features * self.nz, self.ny, self.nx)
        return batch_dict


class AiMotiveRadarDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.split = dataset_cfg.get("TRAIN_SPLIT", "train") if training else dataset_cfg.get("EVAL_SPLIT", "val")
        self.sweeps = dataset_cfg.get("SWEEPS", SWEEPS)
        self.motion = dataset_cfg.get("MOTION", False)  # Phase 6b D2: Doppler motion correction of older sweeps
        self.clutter = dataset_cfg.get("CLUTTER", None)  # Phase 6b D2: path of the clutter classifier (car probability feature)
        self._clf = None
        self.items = splits.frames(self.split)
        if logger:
            logger.info(f"aiMotive radar {self.split}: {len(self.items)} frames, {self.sweeps} sweeps")

    def set_split(self, split):
        self.split = split
        self.items = splits.frames(split)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        seq, frame = self.items[index]
        points = radar_points(seq, frame, self.sweeps, self.motion)
        if self.clutter:
            from clocs_r import radar_clutter
            if self._clf is None:  # loaded in each data-loader worker
                self._clf = radar_clutter.load(self.clutter)
            points = np.concatenate([points, radar_clutter.predict(self._clf, points)[:, None]], 1).astype(np.float32)
        input_dict = dict(points=points, frame_id=f"{seq.path}|{frame}")
        if self.training:
            boxes = radar_labels(seq, frame, points[:, :len(FEATURES)])
            input_dict.update(gt_boxes=boxes, gt_names=np.array(["Car"] * len(boxes)))
        return self.prepare_data(input_dict)
