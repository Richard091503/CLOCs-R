"""Score the CLOCs authors' KITTI models for comparison with aiMotive: SECOND, Cascade R-CNN detections, and CLOCs fusion.

SECOND and CLOCs get the official KITTI metric, to check against the CLOCs README, and all three get the same evaluator as
aiMotive (clocs_r.metrics). Run from second/, since the CLOCs config uses relative paths:

    cd second && python -m clocs_r.eval_kitti --out ../results/detectors
    cd second && python -m clocs_r.eval_kitti --out ../results/clocs_no_radar --only clocs
"""
import argparse
import json
import pathlib
import pickle

import numpy as np
import torch
import torchplus
from google.protobuf import text_format

from clocs_r import metrics
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import box_coder_builder, input_reader_builder
from second.pytorch.models import fusion
from second.pytorch.train import build_inference_net, example_convert_to_torch, predict_v2
from second.utils.eval import get_official_eval_result

CONFIG = "./configs/car.fhd.config"
FUSION_CKPT = "../CLOCs_SecCas_pretrained/fusion_layer-11136.tckpt"
IGNORE = ("Van", "Truck", "Tram")  # like large vehicles on aiMotive: neither targets nor false positives
RANGE_X, RANGE_Y = 70.4, 40.0  # SECOND's KITTI grid


def cam_to_metric(loc, dims, ry):
    """KITTI camera boxes (bottom center, dims l, h, w) -> clocs_r boxes (x forward, y left, z up, center, l, w, h, yaw)."""
    return np.stack([loc[:, 2], -loc[:, 0], -loc[:, 1] + dims[:, 1] / 2, dims[:, 0], dims[:, 2], dims[:, 1], -ry - np.pi / 2], 1)


def in_grid(boxes):
    return (boxes[:, 0] >= 0) & (boxes[:, 0] < RANGE_X) & (np.abs(boxes[:, 1]) < RANGE_Y)


def build_loader(info_path=None):
    """The unaugmented evaluation reader, on the val frames or on another info file."""
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(CONFIG) as f:
        text_format.Merge(f.read(), config)
    if info_path:
        config.eval_input_reader.kitti_info_path = info_path
    model_cfg = config.model.second
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner = target_assigner_builder.build(model_cfg.target_assigner, bv_range, box_coder)
    dataset = input_reader_builder.build(config.eval_input_reader, model_cfg, training=True,
                                         voxel_generator=voxel_generator, target_assigner=target_assigner)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
                                         collate_fn=merge_second_batch)
    return dataset, loader, config.train_config.detection_2d_path, model_cfg.post_center_limit_range


def in_image(pred, shape, limit):
    """The filters CLOCs applies to its own output: box overlaps the image, center inside the limit range."""
    bbox = pred["bbox"].cpu().numpy()
    lidar = pred["box3d_lidar"].cpu().numpy()
    limit = np.array(limit)
    return (bbox[:, 0] <= shape[1]) & (bbox[:, 1] <= shape[0]) & (bbox[:, 2] >= 0) & (bbox[:, 3] >= 0) & \
           (lidar[:, :3] >= limit[:3]).all(1) & (lidar[:, :3] <= limit[3:]).all(1)


def to_anno(pred, shape, limit, idx):
    mask = in_image(pred, shape, limit)
    bbox = pred["bbox"].cpu().numpy()[mask]
    bbox[:, 2:] = np.minimum(bbox[:, 2:], np.array(shape)[::-1])
    bbox[:, :2] = np.maximum(bbox[:, :2], 0)
    cam = pred["box3d_camera"].cpu().numpy()[mask]
    lidar = pred["box3d_lidar"].cpu().numpy()[mask]
    n = len(cam)
    return dict(name=np.array(["Car"] * n), truncated=np.zeros(n), occluded=np.zeros(n, int),
                alpha=-np.arctan2(-lidar[:, 1], lidar[:, 0]) + cam[:, 6], bbox=bbox, dimensions=cam[:, 3:6], location=cam[:, :3],
                rotation_y=cam[:, 6], score=pred["scores"].cpu().numpy()[mask], image_idx=np.full(n, idx, dtype=np.int64))


def gt_boxes(annos, names):
    mask = np.isin(annos["name"], names)
    return cam_to_metric(annos["location"][mask], annos["dimensions"][mask], annos["rotation_y"][mask])


@torch.no_grad()
def eval_3d(out, name):
    """name: "second" for SECOND's own scores, "clocs" for scores from the authors' pretrained fusion layer."""
    dataset, loader, det2d_path, limit = build_loader()
    net = build_inference_net(CONFIG, "../model_dir")
    fusion_layer = None
    if name == "clocs":
        fusion_layer = fusion.fusion().cuda()
        torchplus.train.restore(FUSION_CKPT, fusion_layer)
        fusion_layer.eval()
    infos = dataset.dataset.kitti_infos
    official_score, official_iou = net._nms_score_threshold, net._nms_iou_threshold
    official, frames = [], []
    for i, example in enumerate(loader):
        example = example_convert_to_torch(example, torch.float32)
        _, preds, _, fusion_input, tensor_index = net(example, det2d_path)
        if fusion_layer is not None:
            fused, _ = fusion_layer(fusion_input.cuda(), tensor_index.cuda())
            preds = dict(preds, cls_preds=fused.reshape(1, 200, 176, 2))
        shape = example["image_shape"][0]

        net._nms_score_threshold, net._nms_iou_threshold = official_score, official_iou
        pred = predict_v2(net, example, preds)[0]
        official.append(to_anno(pred, shape, limit, example["image_idx"][0]))

        # Same conditions as aiMotive: no score threshold, BEV NMS at IoU 0.1
        net._nms_score_threshold, net._nms_iou_threshold = 1e-3, 0.1
        pred = predict_v2(net, example, preds)[0]
        anno = to_anno(pred, shape, limit, example["image_idx"][0])
        det = cam_to_metric(anno["location"], anno["dimensions"], anno["rotation_y"])
        score = anno["score"]
        annos = infos[i]["annos"]
        gt, ignore = gt_boxes(annos, ("Car",)), gt_boxes(annos, IGNORE)
        gt = gt[in_grid(gt)]
        keep = in_grid(det)
        if len(ignore) and len(det):
            keep &= metrics.iou_bev(ignore, det).max(0) < 0.1
        frames.append((gt, det[keep], score[keep]))
        if (i + 1) % 400 == 0:
            print(f"{name} {i + 1}/{len(infos)}", flush=True)
    text = get_official_eval_result([info["annos"] for info in infos], official, ("Car",))
    with open(out / f"kitti_{name}_official.txt", "w") as f:
        f.write(str(text))
    print(text, flush=True)
    with open(out / f"kitti_{name}.json", "w") as f:
        json.dump({"val": {"frames": len(infos), "overall": metrics.evaluate(frames)}}, f, indent=1)


def eval_cascade(out, infos_path, det2d_path):
    with open(infos_path, "rb") as f:
        infos = pickle.load(f)
    frames = []
    for info in infos:
        annos = info["annos"]
        with open(pathlib.Path(det2d_path) / f"{info['image_idx']:06d}.txt") as f:
            rows = [line.split() for line in f if line.startswith("Car ")]
        det = np.array([[float(v) for v in r[4:8]] + [float(r[15])] for r in rows], np.float32).reshape(-1, 5)
        car = annos["name"] == "Car"
        loc = annos["location"]
        frames.append(dict(gt=annos["bbox"][car].astype(np.float32), gt_range=np.hypot(loc[car, 0], loc[car, 2]),
                           ignore=annos["bbox"][np.isin(annos["name"], IGNORE + ("DontCare",))].astype(np.float32),
                           cand=det[:, :4], cand_scores=det[:, 4]))
    result = metrics.evaluate_2d(frames)
    with open(out / "kitti_cascade.json", "w") as f:
        json.dump({"val": {"frames": len(infos), "overall": result}}, f, indent=1)
    print("Cascade AP", result["ap"], "\nCascade candidate recall", result["recall"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/detectors")
    ap.add_argument("--infos", default=str(pathlib.Path.home() / "Documents/KITTI_DATASET_ROOT/kitti_infos_val.pkl"))
    ap.add_argument("--det2d", default="../d2_detection_data")
    ap.add_argument("--only", choices=["second", "cascade", "clocs"])
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "cascade"):
        eval_cascade(out, args.infos, args.det2d)
    if args.only in (None, "second"):
        eval_3d(out, "second")
    if args.only == "clocs":
        eval_3d(out, "clocs")


if __name__ == "__main__":
    main()
