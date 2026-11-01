"""Cascade R-CNN on aiMotive with MMDetection: labels from projected 3D boxes, training, and 2D candidate export.

Usage:
    python -m clocs_r.cascade_2d train  --out runs/cascade
    python -m clocs_r.cascade_2d export --out runs/cascade --ckpt runs/cascade/latest.pth --candidates runs/candidates2d
"""
import argparse
import copy
import json
import os
import pathlib

import mmcv
import mmdet
import numpy as np
import torch
from mmcv import Config
from mmdet.apis import inference_detector, init_detector, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

from clocs_r import aimotive, splits

BASE = os.path.join(os.path.dirname(mmdet.__file__), ".mim/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py")
COCO_WEIGHTS = ("https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/"
                "cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth")
K_2D = 200  # candidates kept per image, as in CLOCs


def write_coco(path, split):
    images, anns = [], []
    for seq, frame in splits.frames(split):
        w, h = seq.meta["image_size"]
        img_id = len(images)
        images.append(dict(id=img_id, file_name=seq.image_path(frame), width=w, height=h))
        for classes, crowd in ((splits.CAR_CLASSES, 0), (splits.IGNORE_CLASSES, 1)):  # large vehicles: neither pos nor neg
            boxes, _ = seq.boxes(frame, classes)
            rect, ok = aimotive.project_boxes(seq, boxes)
            for x0, y0, x1, y1 in rect[ok].tolist():
                anns.append(dict(id=len(anns) + 1, image_id=img_id, category_id=1, iscrowd=crowd,
                                 bbox=[x0, y0, x1 - x0, y1 - y0], area=(x1 - x0) * (y1 - y0)))
    with open(path, "w") as f:
        json.dump(dict(images=images, annotations=anns, categories=[dict(id=1, name="Car")]), f)
    return len(images), len(anns)


def build_cfg(out, epochs, ann_file=None):
    cfg = Config.fromfile(BASE)
    for head in cfg.model.roi_head.bbox_head:
        head.num_classes = 1
    # RPN anchors from 16 px instead of 32: cars beyond 100 m are only about 10-25 px wide here
    cfg.model.rpn_head.anchor_generator.scales = [4]
    # Candidates before NMS with a very low score threshold, top K per image
    cfg.model.test_cfg.rcnn = mmcv.ConfigDict(score_thr=0.001, nms=dict(type="nms", iou_threshold=0.95), max_per_img=K_2D)
    norm = cfg.img_norm_cfg
    size = (1280, 704)
    train_pipeline = [dict(type="LoadImageFromFile"), dict(type="LoadAnnotations", with_bbox=True),
                      dict(type="Resize", img_scale=size, keep_ratio=True), dict(type="RandomFlip", flip_ratio=0.5),
                      dict(type="Normalize", **norm), dict(type="Pad", size_divisor=32),
                      dict(type="DefaultFormatBundle"), dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"])]
    test_pipeline = [dict(type="LoadImageFromFile"),
                     dict(type="MultiScaleFlipAug", img_scale=size, flip=False, transforms=[
                         dict(type="Resize", keep_ratio=True), dict(type="RandomFlip"), dict(type="Normalize", **norm),
                         dict(type="Pad", size_divisor=32), dict(type="ImageToTensor", keys=["img"]),
                         dict(type="Collect", keys=["img"])])]
    common = dict(type="CocoDataset", classes=("Car",), img_prefix="")
    cfg.data = dict(samples_per_gpu=4, workers_per_gpu=4,
                    train=dict(ann_file=ann_file, pipeline=train_pipeline, filter_empty_gt=True, **common),
                    val=dict(ann_file=ann_file, pipeline=test_pipeline, test_mode=True, **common),
                    test=dict(ann_file=ann_file, pipeline=test_pipeline, test_mode=True, **common))
    cfg.optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001)  # linear scaling: 0.02 at batch 16
    cfg.optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    cfg.lr_config = dict(policy="step", warmup="linear", warmup_iters=300, warmup_ratio=0.001, step=[epochs - 1])
    cfg.runner = dict(type="EpochBasedRunner", max_epochs=epochs)
    cfg.checkpoint_config = dict(interval=1, max_keep_ckpts=1)
    cfg.log_config = dict(interval=100, hooks=[dict(type="TextLoggerHook")])
    cfg.evaluation = dict(interval=1000, metric="bbox")
    cfg.load_from = COCO_WEIGHTS
    cfg.resume_from = None
    cfg.work_dir = str(out)
    cfg.gpu_ids = [0]
    cfg.device = "cuda"
    cfg.seed = 3
    cfg.workflow = [("train", 1)]
    return cfg


def train(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ann = out / "train_coco.json"
    if not ann.exists():
        print("COCO labels:", write_coco(ann, "train"), flush=True)
    cfg = build_cfg(out, args.epochs, str(ann))
    cfg.dump(str(out / "config.py"))
    dataset = build_dataset(cfg.data.train)
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    model.CLASSES = dataset.CLASSES
    train_detector(model, [dataset], cfg, distributed=False, validate=False, meta=dict(seed=cfg.seed, exp_name="cascade"))


def export(args):
    cfg = build_cfg(args.out, 1, "")
    cfg.load_from = None
    cfg.model.pretrained = None
    model = init_detector(cfg, args.ckpt, device="cuda:0")
    out_root = pathlib.Path(args.candidates)
    for split in args.splits.split(","):
        items = splits.frames(split)
        for i, (seq, frame) in enumerate(items):
            path = out_root / seq.name / f"{frame:07d}.npy"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            det = inference_detector(model, seq.image_path(frame))[0]  # (K, 5): x0, y0, x1, y1, score
            np.save(path, det.astype(np.float32))
            if i % 500 == 0:
                print(split, i, len(items), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "export"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--ckpt")
    ap.add_argument("--candidates", default="runs/candidates2d")
    ap.add_argument("--splits", default="val,holdout,train")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    (train if args.mode == "train" else export)(args)


if __name__ == "__main__":
    main()
