"""Validate the CLOCs-R fusion code on KITTI: does it reproduce the gain of the authors' CLOCs over SECOND?

Uses the authors' SECOND and 2D detections, and otherwise the same code as aiMotive: top-K candidates, cells, targets, network,
loss, NMS and evaluator. Run from second/:

    cd second && python -m clocs_r.kitti_check --out ../results/clocs_no_radar
"""
import argparse
import json
import pathlib
import pickle

import numpy as np
import torch

from clocs_r import fusion, metrics, train_fusion
from clocs_r.cache_cells import NEG_IOU, POS_IOU
from clocs_r.eval_kitti import CONFIG, IGNORE, build_loader, cam_to_metric, gt_boxes, in_grid
from clocs_r.export_3d import K_3D
from second.pytorch.train import build_inference_net, example_convert_to_torch

MAX_RANGE = 82.0  # normalizes distance, as in the CLOCs code
INFOS = pathlib.Path.home() / "Documents/KITTI_DATASET_ROOT"
SEEDS = (1, 2, 3)


@torch.no_grad()
def build_items(net, split):
    dataset, loader, det2d_path, limit = build_loader(str(INFOS / f"kitti_infos_{split}.pkl"))
    infos = dataset.dataset.kitti_infos
    limit = np.array(limit)
    items = []
    for i, example in enumerate(loader):
        example = example_convert_to_torch(example, torch.float32)
        res, _, top_2d, _, _ = net(example, det2d_path)
        res = res[0]
        scores_all = res["scores"].reshape(-1)
        top = torch.topk(scores_all, min(K_3D, len(scores_all))).indices
        scores = scores_all[top].cpu().numpy().astype(np.float32)
        cam = res["box3d_camera"][top].cpu().numpy()
        lidar = res["box3d_lidar"][top].cpu().numpy()
        proj = res["bbox"][top].cpu().numpy().astype(np.float32)  # already clipped to the image
        boxes = cam_to_metric(cam[:, :3], cam[:, 3:6], cam[:, 6]).astype(np.float32)
        dist = (np.hypot(lidar[:, 0], lidar[:, 1]) / MAX_RANGE).astype(np.float32)
        cells, column = fusion.build_cells(top_2d[:, :4].astype(np.float32), top_2d[:, 4].astype(np.float32), proj, scores, dist,
                                           np.zeros((len(scores), 0), np.float32))

        annos = infos[i]["annos"]
        gt = gt_boxes(annos, ("Car",)).astype(np.float32)
        ignore = gt_boxes(annos, IGNORE).astype(np.float32)
        gt = gt[in_grid(gt)]
        target = np.zeros(len(scores), np.float32)
        weight = np.ones(len(scores), np.float32)
        best = metrics.iou_bev(gt, boxes).max(0) if len(gt) else np.zeros(len(scores))
        target[best >= POS_IOU] = 1.0
        weight[(best > NEG_IOU) & (best < POS_IOU)] = 0.0
        visible = ((proj[:, 2] - proj[:, 0]) > 0) & ((proj[:, 3] - proj[:, 1]) > 0)
        keep_eval = visible & in_grid(boxes) & (lidar[:, :3] >= limit[:3]).all(1) & (lidar[:, :3] <= limit[3:]).all(1)
        if len(ignore):
            on_ignored = metrics.iou_bev(ignore, boxes).max(0) >= 0.1
            weight[on_ignored] = 0.0
            keep_eval &= ~on_ignored
        items.append(dict(pairs=cells[:, :2].astype(np.float32), column=column.astype(np.int16), dist=dist, boxes=boxes, scores=scores,
                          target=target, weight=weight, gt=gt, keep_eval=keep_eval, seq=str(infos[i]["image_idx"]), odd="kitti"))
        if (i + 1) % 500 == 0:
            print(f"{split} {i + 1}/{len(infos)}", flush=True)
    return items


def score(net, items, device):
    return metrics.evaluate([(it["gt"], b, s) for it, b, s in train_fusion.predict(net, items, [None] * len(items), [], device)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/clocs_no_radar")
    ap.add_argument("--cache", default="../runs/kitti_cells")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    device = torch.device("cuda")
    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    data = {}
    net = None
    for split in ("train", "val"):
        path = cache / f"{split}.pkl"
        if not path.exists():
            net = net or build_inference_net(CONFIG, "../model_dir")
            with open(path, "wb") as f:
                pickle.dump(build_items(net, split), f, protocol=4)
        with open(path, "rb") as f:
            data[split] = pickle.load(f)
    del net
    positives = np.mean([it["target"].sum() for it in data["train"]])
    print(f"train {len(data['train'])} frames, val {len(data['val'])} frames, {positives:.1f} positives per frame", flush=True)
    results = {"second_same_path": score(None, data["val"], device), "seeds": {}}
    for seed in SEEDS:
        net, log = train_fusion.fit(data["train"], [None] * len(data["train"]), [], args.epochs, seed, device, f"seed {seed}")
        results["seeds"][str(seed)] = dict(val=score(net, data["val"], device), final_loss=log[-1]["loss"], loss_every_1000=[l["loss"] for l in log if l["step"] % 1000 == 0])
        v = results["seeds"][str(seed)]["val"]
        print(f"seed {seed}", {b: round(v[b]["APBEV@0.3"], 3) for b in v}, flush=True)
    with open(pathlib.Path(args.out) / "kitti_our_clocs.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
