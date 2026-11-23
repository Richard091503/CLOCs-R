"""Review round, task 3: time per frame of every LCR step, and parameter counts.

Batch size 1, val frames spread evenly over the split: 20 warm-up frames, then N timed frames. CUDA is synchronized
before and after each timed step. Each detector runs in its own process (--part), because OpenPCDet merges every
loaded config into one global object.

    cd ~/Documents/CLOCs-R && source env.sh
    for p in second d2 cascade lcr params; do python ../CLOCs-R-review/results/review_round/timing.py --part $p --out ../CLOCs-R-review/results/review_round/timing_$p.json; done

Steps (where each runs):
- second: input (read LiDAR file, voxelize; CPU) and network (to the top 2,000 boxes before NMS, as export_3d; GPU).
- d2: input (5 radar sweeps, ego and Doppler motion correction, clutter MLP, pillars; CPU), network (GPU) and its
  output filter (score >= 0.01, BEV NMS 0.1, at most 100, as export_3d --nms; CPU).
- cascade: mmdet inference_detector at the 1.5x image size (read JPEG, resize, normalize on CPU; network on GPU).
- lcr: LCR cells from the three detectors' outputs (projection, camera pairs, K40 floor, radar pairs, reserved cells;
  CPU), the fusion forward pass with max and sigmoid (GPU), and BEV NMS of the fused candidates (CPU). The detector
  outputs are read from the caches before the clock starts. The cells are checked equal to the cached LCR cells.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from clocs_r import splits

N_WARM = 20
CKPT = dict(second="runs/lcr/det/second/checkpoint_epoch_40.pth", d2="runs/lcr/d2/checkpoint_epoch_40.pth",
            cascade="runs/lcr/det/cascade/epoch_4.pth", clutter="runs/lcr/clutter.pt",
            lcr="results/lcr/weights/R0+T2+K40+RC_s1.pth")
D2_CFG = "clocs_r/configs/pointpillar_radar_d2_aimotive.yaml"


def sample(n):
    frames = splits.frames("val")
    return [frames[i] for i in np.linspace(0, len(frames) - 1, N_WARM + n).astype(int)]


class Clock:
    def __init__(self):
        self.t = {}

    def __call__(self, name, fn, record=True):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        if record:
            self.t.setdefault(name, []).append(1000 * (time.perf_counter() - t0))
        return out

    def summary(self):
        return {k: dict(n=len(v), median_ms=float(np.median(v)), p90_ms=float(np.percentile(v, 90)), mean_ms=float(np.mean(v)))
                for k, v in self.t.items()}


def pcdet_part(items, which, clock):
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu

    from clocs_r import metrics
    from clocs_r.train_second import CONFIG, load_cfg
    c = load_cfg(CONFIG if which == "second" else D2_CFG)
    ds, _, _ = build_dataloader(c.DATA_CONFIG, c.CLASS_NAMES, 1, dist=False, workers=0, training=False)
    ds.set_split("val")
    ds.items = items
    model = build_network(c.MODEL, len(c.CLASS_NAMES), ds).cuda()
    model.load_state_dict(torch.load(CKPT[which], map_location="cpu")["model_state"])
    model.eval()
    for i in range(len(items)):
        rec = i >= N_WARM
        batch = clock(f"{which} input (CPU)", lambda: ds.collate_batch([ds[i]]), rec)

        @torch.no_grad()
        def net():
            load_data_to_gpu(batch)
            b = batch
            for m in model.module_list:
                b = m(b)
            sc = torch.sigmoid(b["batch_cls_preds"][0, :, 0])
            top = torch.topk(sc, 2000)
            return b["batch_box_preds"][0, top.indices].cpu().numpy(), top.values.cpu().numpy()
        boxes, sc = clock(f"{which} network (GPU)", net, rec)
        if which == "d2":
            def filt():
                bb, ss = boxes[sc >= 0.01], sc[sc >= 0.01]
                return metrics.nms_bev(bb, ss, 0.1, pre_max=1000, post_max=100)
            clock("d2 output filter: score >= 0.01, BEV NMS (CPU)", filt, rec)
    return sum(p.numel() for p in model.parameters())


def cascade_part(items, clock, scratch):
    from mmdet.apis import inference_detector, init_detector

    from clocs_r import cascade_2d
    cfg = cascade_2d.build_cfg(scratch, 1, "", cascade_2d.SCALE)
    cfg.load_from = None
    cfg.model.pretrained = None
    model = init_detector(cfg, CKPT["cascade"], device="cuda:0")
    for i, (seq, f) in enumerate(items):
        clock(f"cascade R-CNN at {cascade_2d.SCALE}x (image read and resize on CPU, network on GPU)",
              lambda: inference_detector(model, seq.image_path(f))[0], i >= N_WARM)
    return sum(p.numel() for p in model.parameters()), cfg.data.test.pipeline[1]["img_scale"]


def lcr_part(items, clock):
    import pickle

    from clocs_r import aimotive, cache_cells, fusion, metrics, radar_pairs, visibility
    from clocs_r import tune_fusion as tf
    from clocs_r.train_fusion import frame_cells
    cfg = tf.parse("R0+T2+K40+RC")
    state = torch.load(CKPT["lcr"], map_location="cpu")
    assert state["settings"] == cfg
    net = tf.make_net(cfg).cuda()
    net.load_state_dict(state["model"])
    net.eval()
    # the cached LCR cells of these frames, for the equality check
    base, _ = cache_cells.load("val", "runs/lcr/cells", *tf.GATE, cache_cells.SWEEPS, use_radar=False)
    key = {(it["seq"], it["frame"]): k for k, it in enumerate(base)}
    rp = radar_pairs.build("val", "runs/lcr/cells", "runs/lcr/d2_dets", base)
    top2d = visibility.load("val", "runs/lcr/cells", base)["top2d"]
    checked = 0
    for i, (seq, f) in enumerate(items):
        rec = i >= N_WARM
        boxes, scores, det2d = cache_cells.load_candidates(seq, f, "runs/lcr/cand/3d", "runs/lcr/cand/2d")
        with np.load(pathlib.Path("runs/lcr/d2_dets") / seq.name / f"{f:07d}.npz") as z:
            rb, rs = z["boxes"], z["scores"]

        def cells():
            proj, valid = aimotive.project_boxes(seq, boxes)
            proj = proj.astype(np.float32)
            proj[~valid] = np.nan
            dist = np.hypot(boxes[:, 0], boxes[:, 1]) / cache_cells.MAX_RANGE
            c, column = fusion.build_cells(det2d[:, :4], det2d[:, 4], proj, scores, dist, np.zeros((len(scores), 0), np.float32))
            item = dict(column=column.astype(np.int16), pairs=c[:, :2].astype(np.float16), scores=scores, dist=dist.astype(np.float32))
            col, pairs = radar_pairs.combined_cells(item, radar_pairs.frame_pairs(boxes, rb, rs), visibility.top_score(det2d[:, 4]))
            return dict(item, column=col, pairs=pairs)
        item = clock("LCR cells: camera pairs with K40, radar pairs, reserved cells (CPU)", cells, rec)
        k = key[(seq.name, f)]
        ref_col, ref_pairs = radar_pairs.combined_cells(base[k], rp[k], top2d[k])
        assert np.array_equal(item["column"], ref_col) and np.array_equal(item["pairs"], ref_pairs), (seq.name, f)
        checked += 1

        @torch.no_grad()
        def forward():
            x, col = frame_cells(item, "cuda")
            return torch.sigmoid(net(x, col, len(scores))).cpu().numpy()
        fused = clock("LCR fusion network: cells, max per candidate, residual, sigmoid (GPU)", forward, rec)
        clock("BEV NMS of the fused candidates, IoU 0.1, at most 100 (CPU)", lambda: metrics.nms_bev(boxes, fused, 0.1), rec)
    return sum(p.numel() for p in net.parameters()), checked


def params():
    from clocs_r import radar_clutter
    out = {}
    for k in ("second", "d2"):
        out[k] = sum(v.numel() for v in torch.load(CKPT[k], map_location="cpu")["model_state"].values())
    out["cascade"] = sum(v.numel() for v in torch.load(CKPT["cascade"], map_location="cpu")["state_dict"].values())
    out["clutter MLP"] = sum(p.numel() for p in radar_clutter.load(CKPT["clutter"])[0].parameters())
    out["LCR fusion"] = sum(v.numel() for v in torch.load(CKPT["lcr"], map_location="cpu")["model"].values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["second", "d2", "cascade", "lcr", "params"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    clock, extra = Clock(), {}
    if args.part == "params":
        extra["state_dict_numel"] = params()
    else:
        items = sample(args.n)
        extra["frames"] = dict(warmup=N_WARM, timed=args.n, split="val", first=f"{items[0][0].name}|{items[0][1]}")
        if args.part in ("second", "d2"):
            extra["model_parameters"] = pcdet_part(items, args.part, clock)
        elif args.part == "cascade":
            extra["model_parameters"], extra["img_scale"] = cascade_part(items, clock, str(pathlib.Path(args.out).parent / "_cascade_cfg"))
        else:
            extra["model_parameters"], extra["cells_checked_equal_to_cache"] = lcr_part(items, clock)
    extra["torch_threads"] = torch.get_num_threads()
    res = dict(part=args.part, steps=clock.summary(), **extra)
    json.dump(res, open(args.out, "w"), indent=1, default=str)
    print(json.dumps(res, default=str))


if __name__ == "__main__":
    main()
