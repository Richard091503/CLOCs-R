"""Post hoc tuning of the fusion stage, following results/clocs_radar/tuning/PLAN.md. Reads holdout only unless --splits says otherwise.

A configuration joins changes with "+": a radar feature set (R0 for none, R1-R7 of radar_features.FEATURE_SETS; R2 if none is
named), T1 (far targets), T2 (residual),
T3 (logit input), T4a or T4b (5 or 10 sweeps with motion correction) and T5 (RCS). --far_weight applies the range-weighted loss
(range_weight.py) to every configuration of a run. --gate sets the radar gate (delta0,kappa); --nocam scores saved networks
(--from_weights) with every 2D candidate removed.

Usage: python -m clocs_r.tune_fusion --configs R2,T1,T2 --out results/clocs_radar/tuning/selection.json
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import pickle
import time

import numpy as np
import torch

from clocs_r import cache_cells, fusion, metrics, radar_candidates, radar_features, splits, train_fusion, visibility

GATE = (0.5, 0.02)
BASE_SET = "R2"
FAR_FROM = 50.0  # m
FAR_POS, FAR_NEG = 0.5, 0.3  # BEV IoU thresholds beyond FAR_FROM for T1
RCS_INDEX = len(radar_features.FEATURES)  # the RCS channel follows the five features
SEEDS = (1, 2, 3)
DATA = {}
STAGGER_S = 20  # s between the CUDA start-ups of parallel seeds: simultaneous start-ups twice failed on the shared GPU (2026-11-20)
# Phase 3 tokens (results/next_round/PLAN.md). A configuration without any of them parses exactly as before.
P3_TOKENS = {"H": ("vis", "height"), "P": ("vis", "points"), "I1": ("min_iou", 0.1), "I2": ("min_iou", 0.25),
             "K40": ("top2d", visibility.TOP_2D), "D2": ("cam_dropout", 0.2), "D3": ("cam_dropout", 0.3),
             "S50": ("oversample_m", 50.0), "S100": ("oversample_m", 100.0)}
P4_TOKENS = ("G", "C")  # Phase 4: G = fusion.GatedFusion, C = driving condition in the gate context
P7_TOKENS = ("NC", "RP", "RC", "DF")  # Phase 7: NC = no camera in training or test; RP = radar-detector boxes as CLOCs' pairs
# (radar_pairs.py); RC = radar-detector pairs beside the K40 camera pairs in one network (radar_pairs.combined_cells);
# DF = with RC, camera cells of candidates beyond DF_M hidden with probability DF_P in training only (fusion.drop_far_camera)
DF_M, DF_P = 50.0, 0.5
P6_TOKENS = ("RD",)  # Phase 6: radar-detector boxes as candidates, and rd and src as radar channels (radar_candidates.py)
P3_GROUPS = ({"I1", "I2", "K40"}, {"D2", "D3"}, {"S50", "S100"})  # alternatives: at most one of each
OVERSAMPLE = 2.0  # S50, S100: frames with a car beyond the range are drawn this many times as often


def parse(config):
    """Turn a name such as T1+T2 into its settings."""
    parts = config.split("+")
    assert set(parts) <= set(radar_features.FEATURE_SETS) | {"T1", "T2", "T3", "T4a", "T4b", "T5"} | set(P3_TOKENS) | set(P4_TOKENS) | set(P6_TOKENS) | set(P7_TOKENS), config
    assert "C" not in parts or "G" in parts, f"{config}: C is part of the gated model"
    sets = [p for p in parts if p in radar_features.FEATURE_SETS]
    assert len(sets) <= 1, config
    sweeps = 5 if "T4a" in parts else 10 if "T4b" in parts else cache_cells.SWEEPS
    base = sets[0] if sets and sets[0] != "R0" else BASE_SET  # R0 keeps the old placeholder, so saved settings still match
    cfg = dict(radar="R0" not in parts, base=base, far_targets="T1" in parts, sweeps=sweeps, motion="T4a" in parts or "T4b" in parts, rcs="T5" in parts,
               net=dict(residual="T2" in parts, logit_input="T3" in parts))
    p3 = [p for p in parts if p in P3_TOKENS]
    if p3:
        assert all(len(g & set(p3)) <= 1 for g in P3_GROUPS), f"{config}: two alternatives of one Phase 3 group"
        cfg["p3"] = dict(vis=[n for n in visibility.FEATURES if any(P3_TOKENS[p] == ("vis", n) for p in p3)],
                         min_iou=0.0, top2d=0, cam_dropout=0.0, oversample_m=0.0, oversample=OVERSAMPLE)
        cfg["p3"].update({P3_TOKENS[p][0]: P3_TOKENS[p][1] for p in p3 if P3_TOKENS[p][0] != "vis"})
    if "G" in parts:
        assert not cfg.get("p3", {}).get("vis"), f"{config}: H and P are already in the gate context"
        cfg["gated"] = dict(condition="C" in parts, moving=visibility.MOVING)
    if "RD" in parts:
        assert cfg["radar"] and not cfg["rcs"], f"{config}: RD needs radar channels and takes the columns T5 uses"
        cfg["rd"] = dict(min_score=radar_candidates.RD_MIN, match_m=radar_candidates.MATCH_M)
    if "NC" in parts:
        assert "RP" not in parts and "RD" not in parts and not cfg.get("p3"), f"{config}: NC removes every pair; nothing to filter"
        cfg["nocam"] = True
    if "RP" in parts:
        assert not cfg.get("p3") and "RD" not in parts and "G" not in parts, f"{config}: RP replaces the camera pairs"
        cfg["rp"] = dict(min_score=radar_candidates.RD_MIN, match_m=radar_candidates.MATCH_M)
    if "RC" in parts:
        assert "K40" in parts and not cfg["radar"] and "RP" not in parts and "NC" not in parts and "RD" not in parts \
            and "G" not in parts and not cfg["p3"]["vis"], f"{config}: RC adds radar-detector pairs to R0+T2+K40 only"
        cfg["rc"] = dict(min_score=radar_candidates.RD_MIN, match_m=radar_candidates.MATCH_M)
    if "DF" in parts:
        assert "RC" in parts, f"{config}: DF hides camera cells beside the radar-detector cells of RC"
        cfg["far_cam_dropout"] = dict(far_m=DF_M, p=DF_P)
    return cfg


def gate_context(cfg, split, base, radar):
    """Per-frame (n, c) context of the gated model: height, points, [radar returns, moving returns], [condition one-hot]."""
    vis, moving = DATA["vis_" + split], DATA.get("moving_" + split)
    out = []
    for i, item in enumerate(base):
        n = len(item["scores"])
        cols = [np.asarray(vis["features"][i], np.float32)]
        if cfg["radar"]:
            cols += [np.asarray(radar[i][:n, :1], np.float32), np.asarray(moving[i][:n, None], np.float32)]
        if cfg.get("rd"):  # the source flag: 1 for a radar-sourced candidate
            cols.append(np.asarray(radar[i][:n, radar_candidates.RD_COLUMNS[1]:radar_candidates.RD_COLUMNS[1] + 1], np.float32))
        if cfg["gated"]["condition"]:
            cols.append(np.broadcast_to(np.eye(len(splits.ODDS), dtype=np.float32)[splits.ODDS.index(item["odd"])], (n, len(splits.ODDS))))
        out.append(np.concatenate(cols, 1).astype(np.float16))
    return out


def context_size(cfg):
    return len(visibility.FEATURES) + 2 * cfg["radar"] + len(splits.ODDS) * cfg["gated"]["condition"] + bool(cfg.get("rd"))


def make_net(cfg, idx):
    if cfg.get("gated"):
        return fusion.GatedFusion(len(idx), context_size(cfg))
    return fusion.FusionNet(len(idx) + n_extra(cfg), **cfg["net"])


def p3_inputs(cfg, split):
    """Per-frame extra channels and pairing of a split for a configuration's Phase 3 tokens; (None, None) without any."""
    p3 = cfg.get("p3")
    if not p3:
        return (DATA["ctx_" + split], None) if cfg.get("gated") else (None, None)
    vis = DATA["vis_" + split]
    extra = DATA["ctx_" + split] if cfg.get("gated") else None
    if p3["vis"]:
        cols = [visibility.FEATURES.index(n) for n in p3["vis"]]
        extra = [f[:, cols] for f in vis["features"]]
    pairing = None
    if p3["min_iou"] or p3["top2d"]:
        assert p3["top2d"] in (0, visibility.TOP_2D), "only the cached top-40 floor exists"
        pairing = [(p3["min_iou"], t if p3["top2d"] else fusion.MISSING) for t in vis["top2d"]]
    if cfg.get("rc"):  # RC applies the K40 floor when it builds its cells (radar_pairs.combined_cells)
        pairing = None
    return extra, pairing


def frame_weights(cfg, train):
    p3 = cfg.get("p3")
    if not p3 or not p3["oversample_m"]:
        return None
    far = np.array([len(it["gt"]) > 0 and np.hypot(it["gt"][:, 0], it["gt"][:, 1]).max() >= p3["oversample_m"] for it in train])
    return np.where(far, p3["oversample"], 1.0)


def n_extra(cfg):
    """Input channels beyond [IoU, 2D score, s3D, distance] and radar point features: Phase 3 visibility features, or the
    two radar-detector places of token RC."""
    return (len(cfg["p3"]["vis"]) if cfg.get("p3") else 0) + 2 * bool(cfg.get("rc"))


def names_idx(cfg):
    if not cfg["radar"]:
        return []
    idx = [radar_features.FEATURES.index(f) for f in radar_features.FEATURE_SETS[cfg.get("base", BASE_SET)]]
    if cfg.get("rd"):
        return idx + list(radar_candidates.RD_COLUMNS)
    return idx + [RCS_INDEX] if cfg["rcs"] else idx


def _far_target(job):
    path, frame, cand3d, cand2d = job
    seq = cache_cells._seq(path)
    boxes, scores, _ = cache_cells.load_candidates(seq, frame, cand3d, cand2d)
    gt, _, on_ignored = cache_cells.eval_labels(seq, frame, boxes)
    best = metrics.iou_bev(gt, boxes).max(0) if len(gt) else np.zeros(len(scores))
    far = np.hypot(boxes[:, 0], boxes[:, 1]) >= FAR_FROM
    pos, neg = np.where(far, FAR_POS, cache_cells.POS_IOU), np.where(far, FAR_NEG, cache_cells.NEG_IOU)
    target = (best >= pos).astype(np.float32)
    weight = np.ones(len(scores), np.float32)
    weight[(best > neg) & (best < pos)] = 0.0
    weight[on_ignored] = 0.0
    return target, weight


def far_targets(args):
    path = pathlib.Path(args.cells) / f"{args.train_split}_far_targets.pkl"
    if not path.exists():
        jobs = [(s.path, f, args.cand3d, args.cand2d) for s, f in splits.frames(args.train_split)]
        with mp.Pool(14) as pool:
            result = pool.map(_far_target, jobs, chunksize=16)
        with open(path, "wb") as f:
            pickle.dump(result, f, protocol=4)
    with open(path, "rb") as f:
        return pickle.load(f)


def _rcs_item(job):
    path, frame, cand3d, sweeps, motion = job
    seq = cache_cells._seq(path)
    with np.load(pathlib.Path(cand3d) / seq.name / f"{frame:07d}.npz") as z:
        boxes = z["boxes"]
    pts, _, rcs = seq.radar_accumulated_rcs(frame, sweeps, motion)
    return radar_features.candidate_rcs(boxes, pts, rcs, *GATE)


def radar_data(split, cfg, args):
    """The radar array of a split for this configuration, built once and kept on disk."""
    if not cfg["radar"]:
        return None
    cache_cells.build(split, args.cells, args.cand3d, args.cand2d, *GATE, cfg["sweeps"], motion=cfg["motion"])
    radar = cache_cells.load_radar(split, args.cells, *GATE, cfg["sweeps"], cfg["motion"])
    if not cfg["rcs"]:
        return radar
    path = pathlib.Path(args.cells) / f"{split}_rcs_{cache_cells.radar_name(*GATE, cfg['sweeps'], cfg['motion'])}.pkl"
    if not path.exists():
        jobs = [(s.path, f, args.cand3d, cfg["sweeps"], cfg["motion"]) for s, f in splits.frames(split)]
        with mp.Pool(14) as pool:
            rcs = pool.map(_rcs_item, jobs, chunksize=16)
        with open(path, "wb") as f:
            pickle.dump(np.stack(rcs).astype(np.float16), f, protocol=4)
    with open(path, "rb") as f:
        return np.concatenate([radar, pickle.load(f)[..., None]], -1)


def run_seed(job):
    """In a forked child, so that CUDA is first used here and the parent's cached frames are shared."""
    name, cfg, seed, epochs, eval_splits, out, far_weight, from_weights, nocam = job
    time.sleep(STAGGER_S * (seed - 1) % (3 * STAGGER_S))  # start CUDA in the forked children one at a time (see STAGGER_S)
    torch.set_num_threads(1)
    device = torch.device("cuda")
    idx = names_idx(cfg)
    path = pathlib.Path(out) / f"{name}_s{seed}.pth"
    if from_weights:  # score the saved networks again, e.g. on val after selection, instead of retraining them
        state = torch.load(path, map_location="cpu")
        assert state["settings"] == cfg and state.get("far_weight", 1.0) == far_weight and tuple(state.get("gate", GATE)) == tuple(GATE), \
            f"{path} was trained differently"
        net = make_net(cfg, idx).to(device)
        net.load_state_dict(state["model"])
        result = dict(seed=seed, weights=str(path))
    else:
        train = DATA["train_far"] if cfg["far_targets"] else DATA["train"]
        radar = DATA["radar_train"] if idx else [None] * len(train)
        extra, pairing = p3_inputs(cfg, "train")
        p3 = cfg.get("p3") or {}
        net, log = train_fusion.fit(train, radar, idx, epochs, seed, device, f"{name} s{seed}", net_kwargs=cfg["net"], far_weight=far_weight,
                                    extra=extra, pairing=pairing, cam_dropout=1.0 if cfg.get("nocam") else p3.get("cam_dropout", 0.0),
                                    frame_weights=frame_weights(cfg, train), model=lambda: make_net(cfg, idx),
                                    far_cam_dropout=(cfg["far_cam_dropout"]["far_m"] / cache_cells.MAX_RANGE, cfg["far_cam_dropout"]["p"])
                                    if cfg.get("far_cam_dropout") else None)
        torch.save(dict(model=net.state_dict(), config=name, settings=cfg, gate=GATE, far_weight=far_weight), path)
        result = dict(seed=seed, final_loss=log[-1]["loss"] if log else None, loss=log)
    for split in eval_splits:
        extra, pairing = p3_inputs(cfg, split)
        result[split] = train_fusion.evaluate(net, DATA[split], DATA["radar_" + split] if idx else [None] * len(DATA[split]), idx, device,
                                              nocam or bool(cfg.get("nocam")),
                                              extra=extra, pairing=pairing)
    result["params"] = sum(p.numel() for p in net.parameters())
    return result


def main():
    global GATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="holdout")
    ap.add_argument("--weights", default="runs/tuning")
    ap.add_argument("--cells", default="runs/cells")
    ap.add_argument("--cand3d", default="runs/candidates3d")
    ap.add_argument("--cand2d", default="runs/candidates2d")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--far_weight", type=float, default=1.0, help="range-weighted fusion loss, applied to every configuration")
    ap.add_argument("--from_weights", action="store_true", help="evaluate the networks saved in --weights instead of training")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)), help="comma-separated training seeds, one process each")
    ap.add_argument("--gate", default=",".join(map(str, GATE)), help="radar gate delta0,kappa (Eq. 2)")
    ap.add_argument("--train_split", default="train", help="clocs_r.splits name of the fusion training frames")
    ap.add_argument("--nocam", action="store_true", help="with --from_weights: score with all 2D candidates removed")
    ap.add_argument("--radar_dets", default=radar_candidates.DETS, help="exported radar-detector boxes, for RD configurations")
    args = ap.parse_args()
    GATE = tuple(float(x) for x in args.gate.split(","))
    assert not args.nocam or args.from_weights, "--nocam only scores saved networks"
    seeds = [int(s) for s in args.seeds.split(",")]
    eval_splits = args.splits.split(",")
    pathlib.Path(args.weights).mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(args.out)
    results = json.load(open(out)) if out.exists() else {}
    ctx = mp.get_context("fork")

    if not args.from_weights:
        DATA["train"], _ = cache_cells.load(args.train_split, args.cells, *GATE, cache_cells.SWEEPS, use_radar=False)
    for split in eval_splits:
        DATA[split], _ = cache_cells.load(split, args.cells, *GATE, cache_cells.SWEEPS, use_radar=False)
    rd_run = {bool(parse(n).get("rd")) for n in args.configs.split(",")}
    assert len(rd_run) == 1, "RD configurations change the candidates of every split; run them on their own"
    rp_run = {bool(parse(n).get("rp")) for n in args.configs.split(",")}
    assert len(rp_run) == 1, "RP configurations replace the pairs of every split; run them on their own"
    rc_run = {bool(parse(n).get("rc")) for n in args.configs.split(",")}
    assert len(rc_run) == 1, "RC configurations change the pairs of every split; run them on their own"
    for name in args.configs.split(","):
        cfg = parse(name)
        if cfg["far_targets"] and "train_far" not in DATA and not args.from_weights:  # targets only matter for training
            DATA["train_far"] = [dict(item, target=t, weight=w) for item, (t, w) in zip(DATA["train"], far_targets(args))]
        keep = cfg.get("rd") and DATA.get("rd_done_train")  # RD radar arrays are augmented once and kept
        if not keep:
            DATA["radar_train"] = None if args.from_weights else radar_data(args.train_split, cfg, args)
        for split in eval_splits:
            if not (cfg.get("rd") and DATA.get("rd_done_" + split)):
                DATA["radar_" + split] = radar_data(split, cfg, args)
        if cfg.get("p3") or cfg.get("gated"):  # visibility inputs, built once per split beside the cells
            for split, key in [(s, s) for s in eval_splits] + ([] if args.from_weights else [(args.train_split, "train")]):
                if "vis_" + key not in DATA:
                    visibility.build(split, args.cells, args.cand3d, args.cand2d)
                    DATA["vis_" + key] = visibility.load(split, args.cells, DATA[key])
                if cfg.get("gated"):
                    if cfg["radar"] and "moving_" + key not in DATA:
                        DATA["moving_" + key] = visibility.build_moving(split, args.cells, args.cand3d, *GATE, cfg["sweeps"])
                    DATA["ctx_" + key] = gate_context(cfg, key, DATA[key], DATA["radar_" + key])
        if cfg.get("rp"):  # Phase 7: radar-detector pairs take the place of the camera pairs, once per split
            from clocs_r import radar_pairs
            for split, key in [(s, s) for s in eval_splits] + ([] if args.from_weights else [(args.train_split, "train")]):
                if not DATA.get("rp_done_" + key):
                    DATA[key] = radar_pairs.replace_pairs(DATA[key], radar_pairs.build(split, args.cells, args.radar_dets, DATA[key]))
                    DATA["rp_done_" + key] = True
        if cfg.get("rc"):  # Phase 7 LCR: radar-detector pairs beside the K40 camera pairs, once per split (after vis_)
            from clocs_r import radar_pairs
            for split, key in [(s, s) for s in eval_splits] + ([] if args.from_weights else [(args.train_split, "train")]):
                if not DATA.get("rc_done_" + key):
                    DATA[key] = radar_pairs.add_pairs(DATA[key], radar_pairs.build(split, args.cells, args.radar_dets, DATA[key]),
                                                      DATA["vis_" + key]["top2d"])
                    DATA["rc_done_" + key] = True
        if cfg.get("rd"):  # radar-sourced candidates appended to every frame (Phase 6), once per split
            for split, key in [(s, s) for s in eval_splits] + ([] if args.from_weights else [(args.train_split, "train")]):
                if not DATA.get("rd_done_" + key):
                    rd = radar_candidates.build(split, args.cells, args.cand3d, args.cand2d, args.radar_dets)
                    DATA[key], DATA["radar_" + key], DATA["vis_" + key], DATA["moving_" + key] = radar_candidates.augment(
                        DATA[key], DATA["radar_" + key], DATA["vis_" + key], DATA["moving_" + key], rd)
                    del rd
                    DATA["rd_done_" + key] = True
                if cfg.get("gated"):
                    DATA["ctx_" + key] = gate_context(cfg, key, DATA[key], DATA["radar_" + key])
        with ctx.Pool(len(seeds)) as pool:
            runs = pool.map(run_seed, [(name, cfg, s, args.epochs, eval_splits, args.weights, args.far_weight, args.from_weights, args.nocam)
                                       for s in seeds])
        prev = results.get(name)
        if prev and prev.get("settings") == cfg and prev.get("far_weight") == args.far_weight and prev.get("gate") == list(GATE) \
                and prev.get("nocam") == args.nocam:  # another seed group of the same run: keep both, by seed (fixed 2026-11-20)
            runs = sorted({**{r["seed"]: r for r in prev["runs"]}, **{r["seed"]: r for r in runs}}.values(), key=lambda r: r["seed"])
        results[name] = dict(settings=cfg, far_weight=args.far_weight, gate=list(GATE), nocam=args.nocam, runs=runs)
        far = {split: [r[split]["overall"]["50-200"]["APCD"] for r in runs] for split in eval_splits}
        print(name, {k: f"{np.mean(v):.4f} ({[round(x, 4) for x in v]})" for k, v in far.items()}, flush=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
