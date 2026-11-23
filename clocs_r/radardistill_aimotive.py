"""RadarDistill (Bang et al., CVPR 2024; the authors' code at ~/RadarDistill, commit a81d248) on aiMotive
(results/next_round/PLAN.md, Phase 6b, detector D3).

Their pcdet is put first on sys.path, so this module and its callers must be imported before anything imports pcdet.
- stage teacher: their LiDAR PillarNet (configs/radardistill/teacher.yaml) trained on fit.
- stage student: their radar student, initialized from the teacher as their ckpt.py does (each weight copied to a
  "radar_" twin), distilled with the teacher frozen (configs/radardistill/student.yaml).
- export: the student alone (configs/radardistill/student_test.yaml), radar points only, boxes after BEV NMS (0.1, at
  most 100) with score >= 0.01, saved as export_3d --nms saves them.

As run in Phase 7 (results/next_round/phase7_run.sh; schedules halved by the pre-registered 24-hour rule, D2's input):

    python -m clocs_r.radardistill_aimotive --stage teacher --out runs/next_round/phase7/rd_teacher --epochs 10 --radar_input d2
    python -m clocs_r.radardistill_aimotive --stage student --init runs/next_round/phase7/rd_teacher/checkpoint_epoch_10.pth \
        --out runs/next_round/phase7/rd_student --epochs 20 --radar_input d2
    python -m clocs_r.radardistill_aimotive --stage export --ckpt runs/next_round/phase7/rd_student/checkpoint_epoch_20.pth \
        --out runs/next_round/phase7/d3_dets --splits fit,select,val --radar_input d2
"""
import os
import sys

RD_ROOT = os.path.expanduser("~/RadarDistill")
if RD_ROOT not in sys.path:
    sys.path.insert(0, RD_ROOT)
    sys.path.insert(0, os.path.join(RD_ROOT, "tools"))

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict  # noqa: E402

import importlib  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Their pcdet.datasets imports every dataset class, including nuScenes' (nuscenes-devkit) at load time. aiMotive uses
# none of them, so the devkits that are not installed are replaced by empty stand-ins instead of being installed.
_OPTIONAL = ("nuscenes", "kornia", "av2", "waymo_open_dataset", "openpyxl")  # openpyxl: an unused import in their backbone


def _stub(name):
    for i in range(1, name.count(".") + 2):
        part = ".".join(name.split(".")[:i])
        if part not in sys.modules:
            m = types.ModuleType(part)
            m.__path__ = []
            m.__getattr__ = lambda attr: type(attr, (), {})
            sys.modules[part] = m


for _mod in ("pcdet.datasets", "pcdet.models"):
    for _ in range(20):
        try:
            importlib.import_module(_mod)
            break
        except ModuleNotFoundError as e:
            if not e.name or not e.name.split(".")[0] in _OPTIONAL:
                raise
            _stub(e.name)

import pcdet  # noqa: E402
from pcdet import datasets as pcdet_datasets  # noqa: E402
from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.datasets.dataset_distill import DatasetTemplate_Distill  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

assert pcdet.__file__.startswith(RD_ROOT), f"pcdet must be RadarDistill's, got {pcdet.__file__}"

from clocs_r import aimotive, metrics, radar_dataset, splits  # noqa: E402
from clocs_r.second_dataset import points_in_box  # noqa: E402

CONFIGS = pathlib.Path(__file__).parent / "configs" / "radardistill"


def lidar_labels(seq, frame, lidar):
    """SECOND's training labels: cars in the camera view with at least one LiDAR point."""
    boxes, _ = seq.boxes(frame, splits.CAR_CLASSES)
    if len(boxes):
        boxes = boxes[aimotive.in_camera_view(seq, boxes)]
    keep = [points_in_box(lidar, b).any() for b in boxes]
    return boxes[np.array(keep, dtype=bool)] if len(boxes) else boxes


class AiMotiveDistillDataset(DatasetTemplate_Distill):
    """LiDAR in "points" and radar in "radar_points"; with RADAR_ONLY, radar in "points" (their test-time student input)."""

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.split = dataset_cfg.get("TRAIN_SPLIT", "fit") if training else dataset_cfg.get("EVAL_SPLIT", "select")
        self.sweeps = dataset_cfg.get("SWEEPS", radar_dataset.SWEEPS)
        self.motion = dataset_cfg.get("MOTION", False)
        self.clutter = dataset_cfg.get("CLUTTER", None)
        self.radar_only = dataset_cfg.get("RADAR_ONLY", False)
        self._clf = None
        self.items = splits.frames(self.split)
        if logger:
            logger.info(f"aiMotive distill {self.split}: {len(self.items)} frames, radar only {self.radar_only}")

    def set_split(self, split):
        self.split = split
        self.items = splits.frames(split)

    def __len__(self):
        return len(self.items)

    def radar(self, seq, frame):
        pts = radar_dataset.radar_points(seq, frame, self.sweeps, self.motion)
        if self.clutter:
            from clocs_r import radar_clutter
            if self._clf is None:
                self._clf = radar_clutter.load(self.clutter)
            pts = np.concatenate([pts, radar_clutter.predict(self._clf, pts)[:, None]], 1).astype(np.float32)
        return pts

    def __getitem__(self, index):
        seq, frame = self.items[index]
        radar = self.radar(seq, frame)
        if self.radar_only:
            # their test-time student reads radar from "points"; the distill encoder also expects "radar_points"
            return self.prepare_data(dict(points=radar, radar_points=radar.copy(), frame_id=f"{seq.path}|{frame}"))
        lidar = seq.lidar(frame)
        d = dict(points=lidar, radar_points=radar, frame_id=f"{seq.path}|{frame}")
        if self.training:
            boxes = lidar_labels(seq, frame, lidar)
            d.update(gt_boxes=boxes, gt_names=np.array(["car"] * len(boxes)))
        return self.prepare_data(d)


def load_cfg(name):
    pcdet_datasets.__all__["AiMotiveDistillDataset"] = AiMotiveDistillDataset
    cfg_from_yaml_file(str(CONFIGS / name), cfg)
    cfg.LOCAL_RANK = 0
    return cfg


def init_student(teacher_ckpt, out):
    """Their ckpt.py: every teacher weight is kept and copied to a "radar_" twin."""
    t = torch.load(teacher_ckpt, map_location="cpu")
    state = OrderedDict()
    for k, v in t["model_state"].items():
        state[k] = v
        state["radar_" + k] = v
    torch.save(dict(model_state=state, epoch=t["epoch"], it=t["it"]), out)
    return out


def train(stage, out, epochs=None, init=None, workers=6, max_iters=0, radar_input="d1"):
    from train_utils.optimization import build_optimizer, build_scheduler
    c = load_cfg(f"{stage}.yaml")
    set_radar_input(c, radar_input)
    if epochs:
        c.OPTIMIZATION.NUM_EPOCHS = epochs
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    common_utils.set_random_seed(3)
    logger = common_utils.create_logger(out / "train.log", rank=0)
    dataset, loader, _ = build_dataloader(c.DATA_CONFIG, c.CLASS_NAMES, c.OPTIMIZATION.BATCH_SIZE_PER_GPU, dist=False,
                                          workers=workers, logger=logger, training=True, total_epochs=c.OPTIMIZATION.NUM_EPOCHS)
    model = build_network(c.MODEL, len(c.CLASS_NAMES), dataset).cuda()
    if init:
        model.load_params_from_file(str(init), logger=logger, to_cpu=True)
    logger.info(f"parameters: {sum(p.numel() for p in model.parameters())}, trainable "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = build_optimizer(model, c.OPTIMIZATION)
    scheduler, _ = build_scheduler(optimizer, len(loader), c.OPTIMIZATION.NUM_EPOCHS, -1, c.OPTIMIZATION)
    start_epoch, it, losses = 0, 0, []
    ckpts = sorted(out.glob("checkpoint_epoch_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
    if ckpts:
        state = torch.load(ckpts[-1], map_location="cpu")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch, it = state["epoch"] + 1, state["it"]
        losses = json.load(open(out / "loss.json"))
        logger.info(f"resumed from {ckpts[-1].name}")
    model.train()
    t0, it0 = time.time(), it
    for epoch in range(start_epoch, c.OPTIMIZATION.NUM_EPOCHS):
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
                extra = " ".join(f"{k} {v:.3f}" for k, v in tb.items() if isinstance(v, float))[:200]
                logger.info(f"epoch {epoch + 1}/{c.OPTIMIZATION.NUM_EPOCHS} iter {it}/{len(loader) * c.OPTIMIZATION.NUM_EPOCHS} "
                            f"loss {np.mean(losses[-50:]):.3f} {extra} {(it - it0) / (time.time() - t0):.2f} it/s "
                            f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
            if max_iters and it >= max_iters:
                logger.info(f"smoke test done: {(it - it0) / (time.time() - t0):.2f} it/s, {len(loader)} it/epoch")
                return
        torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), epoch=epoch, it=it),
                   out / f"checkpoint_epoch_{epoch + 1}.pth")
        (out / f"checkpoint_epoch_{epoch}.pth").unlink(missing_ok=True)  # only the newest is kept
        json.dump(losses, open(out / "loss.json", "w"))
    logger.info("training finished")


def set_radar_input(c, radar_input):
    """D1's radar points (6 features) or D2's (motion-corrected, plus the clutter classifier's car probability)."""
    feats = list(radar_dataset.FEATURES) + (["car"] if radar_input == "d2" else [])
    c.DATA_CONFIG.MOTION = radar_input == "d2"
    c.DATA_CONFIG.CLUTTER = "runs/next_round/phase6b/clutter.pt" if radar_input == "d2" else None
    enc = c.DATA_CONFIG.POINT_FEATURE_ENCODING
    if "radar_used_feature_list" in enc:
        enc.radar_used_feature_list = enc.radar_src_feature_list = feats
    if c.DATA_CONFIG.get("RADAR_ONLY", False):
        enc.used_feature_list = enc.src_feature_list = feats


@torch.no_grad()
def export(ckpt, out, split_list, radar_input="d1", batch=8, limit=0):
    c = load_cfg("student_test.yaml")
    set_radar_input(c, radar_input)
    dataset, loader, _ = build_dataloader(c.DATA_CONFIG, c.CLASS_NAMES, batch, dist=False, workers=8, training=False)
    model = build_network(c.MODEL, len(c.CLASS_NAMES), dataset).cuda()
    state = torch.load(ckpt, map_location="cpu")["model_state"]
    keep = {k: v for k, v in state.items() if k.startswith("radar_") or k == "global_step"}  # the student, without the teacher
    missing, unexpected = model.load_state_dict(keep, strict=False)
    assert not unexpected and set(missing) <= {"global_step"}, (missing, unexpected)
    model.eval()
    for split in split_list:
        dataset.set_split(split)
        if limit:  # smoke tests only
            dataset.items = dataset.items[:limit]
        t0, n = time.time(), 0
        for b in loader:
            if len(b["points"]) == 0:  # no radar point in any frame of the batch: spconv cannot run on 0 pillars, and
                for fid in b["frame_id"]:  # a frame without radar points has no radar detections
                    path, frame = fid.split("|")
                    dst = pathlib.Path(out) / pathlib.Path(path).name / f"{int(frame):07d}.npz"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(dst, boxes=np.zeros((0, 7), np.float32), scores=np.zeros(0, np.float32))
                n += len(b["frame_id"])
                continue
            load_data_to_gpu(b)
            preds, _ = model(b)
            for i, fid in enumerate(b["frame_id"]):
                path, frame = fid.split("|")
                boxes = preds[i]["pred_boxes"].cpu().numpy().astype(np.float32)[:, :7]
                sc = preds[i]["pred_scores"].cpu().numpy().astype(np.float32)
                boxes, sc = boxes[sc >= 0.01], sc[sc >= 0.01]
                keep = metrics.nms_bev(boxes, sc, 0.1, pre_max=1000, post_max=100)
                dst = pathlib.Path(out) / pathlib.Path(path).name / f"{int(frame):07d}.npz"
                dst.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dst, boxes=boxes[keep], scores=sc[keep])
            n += len(b["frame_id"])
        print(f"{split} done: {n} frames, {n / (time.time() - t0):.1f} frames/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("teacher", "student", "export"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--splits", default="fit,select,val")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max_iters", type=int, default=0)
    ap.add_argument("--radar_input", default="d1", choices=("d1", "d2"))
    ap.add_argument("--limit", type=int, default=0, help="export: only the first N frames of each split (smoke tests)")
    args = ap.parse_args()
    if args.stage == "export":
        export(args.ckpt, args.out, args.splits.split(","), args.radar_input, limit=args.limit)
    elif args.stage == "teacher":
        train("teacher", args.out, args.epochs, max_iters=args.max_iters, radar_input=args.radar_input)
    else:
        init = init_student(args.init, pathlib.Path(args.out).with_suffix(".init.pth")) if args.init else None
        pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
        train("student", args.out, args.epochs, init=init, max_iters=args.max_iters, radar_input=args.radar_input)


if __name__ == "__main__":
    main()
