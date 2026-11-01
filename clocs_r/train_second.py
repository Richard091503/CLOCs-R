"""Train SECOND on aiMotive with OpenPCDet.

Usage: python -m clocs_r.train_second --out runs/second [--epochs N] [--max_iters N]
"""
import argparse
import json
import os
import pathlib
import time

import numpy as np
import torch
from pcdet import datasets as pcdet_datasets
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from clocs_r.second_dataset import AiMotiveDataset

CONFIG = pathlib.Path(__file__).parent / "configs" / "second_aimotive.yaml"


def load_cfg():
    pcdet_datasets.__all__["AiMotiveDataset"] = AiMotiveDataset
    # Relative _BASE_CONFIG_ and ROOT_DIR handling live in pcdet; the config itself is standalone
    cfg_from_yaml_file(str(CONFIG), cfg)
    cfg.LOCAL_RANK = 0
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_iters", type=int, default=0, help="stop early and report speed (smoke test)")
    args = ap.parse_args()

    # Imported here because tools/ is not a package installed with pcdet
    import sys
    sys.path.insert(0, os.path.expanduser("~/OpenPCDet/tools"))
    from train_utils.optimization import build_optimizer, build_scheduler

    c = load_cfg()
    if args.epochs:
        c.OPTIMIZATION.NUM_EPOCHS = args.epochs
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    common_utils.set_random_seed(3)
    logger = common_utils.create_logger(out / "train.log", rank=0)
    dataset, loader, _ = build_dataloader(c.DATA_CONFIG, c.CLASS_NAMES, c.OPTIMIZATION.BATCH_SIZE_PER_GPU, dist=False,
                                          workers=args.workers, logger=logger, training=True,
                                          total_epochs=c.OPTIMIZATION.NUM_EPOCHS)
    model = build_network(c.MODEL, len(c.CLASS_NAMES), dataset).cuda()
    logger.info(f"parameters: {sum(p.numel() for p in model.parameters())}, anchors: {model.dense_head.anchors[0].numel() // 7}")
    optimizer = build_optimizer(model, c.OPTIMIZATION)
    scheduler, _ = build_scheduler(optimizer, len(loader), c.OPTIMIZATION.NUM_EPOCHS, -1, c.OPTIMIZATION)

    model.train()
    it, t0, losses = 0, time.time(), []
    for epoch in range(c.OPTIMIZATION.NUM_EPOCHS):
        loader.dataset.training = True
        for batch in loader:
            scheduler.step(it)
            load_data_to_gpu(batch)
            optimizer.zero_grad()
            ret, tb, _ = model(batch)
            loss = ret["loss"].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.OPTIMIZATION.GRAD_NORM_CLIP)
            optimizer.step()
            it += 1
            losses.append(loss.item())
            if it % 50 == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9
                logger.info(f"epoch {epoch + 1}/{c.OPTIMIZATION.NUM_EPOCHS} iter {it}/{len(loader) * c.OPTIMIZATION.NUM_EPOCHS} loss {np.mean(losses[-50:]):.3f} cls {tb['rpn_loss_cls']:.3f} "
                            f"loc {tb['rpn_loss_loc']:.3f} lr {optimizer.lr:.5f} {it / (time.time() - t0):.2f} it/s mem {mem:.1f}GB")
            if args.max_iters and it >= args.max_iters:
                logger.info(f"smoke test done: {it / (time.time() - t0):.2f} it/s, {len(loader)} it/epoch")
                return
        torch.save(dict(model_state=model.state_dict(), epoch=epoch, it=it), out / f"checkpoint_epoch_{epoch + 1}.pth")
        with open(out / "loss.json", "w") as f:
            json.dump(losses, f)
        for old in sorted(out.glob("checkpoint_epoch_*.pth"), key=os.path.getmtime)[:-2]:
            old.unlink()  # keep the last two; disk is limited
    logger.info("training finished")


if __name__ == "__main__":
    main()
