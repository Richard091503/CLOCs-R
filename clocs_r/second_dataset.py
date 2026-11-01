"""aiMotive as an OpenPCDet dataset, for training and exporting SECOND."""
import numpy as np
from pcdet.datasets import DatasetTemplate

from clocs_r import aimotive, splits


def points_in_box(points, box):
    dx, dy = points[:, 0] - box[0], points[:, 1] - box[1]
    c, s = np.cos(box[6]), np.sin(box[6])
    return (np.abs(dx * c + dy * s) < box[3] / 2) & (np.abs(-dx * s + dy * c) < box[4] / 2) & \
           (np.abs(points[:, 2] - box[2]) < box[5] / 2)


class AiMotiveDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.split = "train" if training else dataset_cfg.get("EVAL_SPLIT", "val")
        self.items = splits.frames(self.split)
        if logger:
            logger.info(f"aiMotive {self.split}: {len(self.items)} frames")

    def set_split(self, split):
        self.split = split
        self.items = splits.frames(split)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        seq, frame = self.items[index]
        points = seq.lidar(frame)
        input_dict = dict(points=points, frame_id=f"{seq.path}|{frame}")
        if self.training:
            boxes, _ = seq.boxes(frame, splits.CAR_CLASSES)
            if len(boxes):
                boxes = boxes[aimotive.in_camera_view(seq, boxes)]
            # LiDAR is the detector's only input, so boxes with no points carry no signal for it
            keep = [points_in_box(points, b).any() for b in boxes]
            boxes = boxes[np.array(keep, dtype=bool)] if len(boxes) else boxes
            input_dict.update(gt_boxes=boxes, gt_names=np.array(["Car"] * len(boxes)))
        return self.prepare_data(input_dict)
