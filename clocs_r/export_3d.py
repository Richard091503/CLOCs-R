"""Run the trained SECOND once over a split and cache its candidates before NMS.

All 32,768 anchor-level boxes for all frames would not fit on this disk, so each frame keeps the K highest-scoring ones.
Usage: python -m clocs_r.export_3d --ckpt runs/second/checkpoint_epoch_40.pth --out runs/candidates3d
"""
import argparse
import pathlib
import time

import numpy as np
import torch
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu

from clocs_r.train_second import load_cfg

K_3D = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="val,holdout,train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    c = load_cfg()
    dataset, loader, _ = build_dataloader(c.DATA_CONFIG, c.CLASS_NAMES, args.batch, dist=False, workers=6, training=False)
    model = build_network(c.MODEL, len(c.CLASS_NAMES), dataset).cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model_state"])
    model.eval()
    out_root = pathlib.Path(args.out)
    for split in args.splits.split(","):
        dataset.set_split(split)
        if args.limit:
            dataset.items = dataset.items[:: max(1, len(dataset.items) // args.limit)][:args.limit]
        t0, n = time.time(), 0
        for batch in loader:
            load_data_to_gpu(batch)
            with torch.no_grad():
                for module in model.module_list:
                    batch = module(batch)
            scores = torch.sigmoid(batch["batch_cls_preds"][..., 0])
            top = torch.topk(scores, K_3D, dim=1)
            for b, fid in enumerate(batch["frame_id"]):
                path, frame = fid.split("|")
                dst = out_root / pathlib.Path(path).name / f"{int(frame):07d}.npz"
                dst.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dst, boxes=batch["batch_box_preds"][b, top.indices[b]].cpu().numpy().astype(np.float32),
                                    scores=top.values[b].cpu().numpy().astype(np.float32))
            n += len(batch["frame_id"])
            if n % 400 == 0:
                print(f"{split} {n}/{len(dataset)} {n / (time.time() - t0):.1f} frames/s", flush=True)
        print(f"{split} done: {n} frames, {n / (time.time() - t0):.1f} frames/s", flush=True)


if __name__ == "__main__":
    main()
