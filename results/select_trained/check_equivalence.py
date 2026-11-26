"""Does master's code train exactly as the thesis code that made the paper's fit-trained networks?

The paper's L0, LRd, LC, CLOCs-R and B were trained with the thesis branch's tune_fusion and selector. The select-trained
models here use master's tune_fusion (with its new NC token and --steps option) and blend.py. This trains each kind of
network on the first N_FRAMES frames of select for EPOCHS epochs with both code bases (seed 1, GPU), and the blend on
fixed logits, and compares the weights. Both packages are named clocs_r, so each side runs in its own process:

    python results/select_trained/check_equivalence.py thesis <dir>   # PYTHONPATH = thesis worktree
    python results/select_trained/check_equivalence.py master <dir>   # PYTHONPATH = this checkout
    python results/select_trained/check_equivalence.py compare <dir> <out.json>
"""
import json
import pathlib
import sys

import numpy as np
import torch

N_FRAMES, EPOCHS, SEED = 200, 2, 1
CELLS, CAND3D, CAND2D = "runs/next_round/phase1/cells_in", "runs/next_round/phase1/cand_in/3d", "runs/next_round/phase1/cand_in/2d"
DETS_THESIS, DETS_MASTER = "runs/next_round/phase7/d2_dets", "runs/lcr/d2_dets"  # the same files (runs/lcr/d2_dets is a link)
MODELS = ("R0+T2+K40", "R0+T2+NC", "R0+T2+RP", "R0+T2+K40+RC")


def fixed_logits(items):
    """Deterministic (n, 3) logits for the blend check: L and two scaled copies."""
    out = []
    for it in items:
        s = np.clip(it["scores"], 1e-6, 1 - 1e-6)
        z = np.log(s / (1 - s)).astype(np.float32)
        out.append(np.stack([z, 0.5 * z + 0.3, 0.2 * z - 0.1], 1).astype(np.float32))
    return out


def thesis(out):
    from clocs_r import cache_cells, radar_pairs, selector, train_fusion, visibility
    from clocs_r import tune_fusion as tf
    device = torch.device("cuda")
    full, _ = cache_cells.load("select", CELLS, *tf.GATE, cache_cells.SWEEPS, use_radar=False)
    top2d = visibility.load("select", CELLS, full)["top2d"][:N_FRAMES]
    rp = radar_pairs.build("select", CELLS, DETS_THESIS, full)[:N_FRAMES]
    base = full[:N_FRAMES]
    del full
    data = {"base": base, "rp": radar_pairs.replace_pairs(base, rp), "rc": radar_pairs.add_pairs(base, rp, top2d)}
    for name in MODELS:
        cfg = tf.parse(name)
        idx = tf.names_idx(cfg)
        items = data["rc" if cfg.get("rc") else "rp" if cfg.get("rp") else "base"]
        pairing = [(0.0, t) for t in top2d] if cfg.get("p3") and not cfg.get("rc") else None
        net, log = train_fusion.fit(items, [None] * len(items), idx, EPOCHS, SEED, device, name, net_kwargs=cfg["net"],
                                    far_weight=2.0, pairing=pairing, cam_dropout=1.0 if cfg.get("nocam") else 0.0,
                                    model=lambda: tf.make_net(cfg, idx))
        torch.save(dict(model=net.state_dict(), settings=cfg, steps=log[-1]["step"] if log else EPOCHS * len(items)),
                   pathlib.Path(out) / f"thesis_{name}.pth")
    ctx = [np.zeros((len(it["scores"]), selector.N_CONTEXT), np.float32) for it in base]
    gate = selector.train_gate(ctx, fixed_logits(base), base, SEED, device, epochs=EPOCHS, kind="blend")
    torch.save(dict(model=gate.state_dict()), pathlib.Path(out) / "thesis_blend.pth")


def master(out):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import blend
    from clocs_r import bootstrap, train_fusion
    from clocs_r import tune_fusion as tf
    device = torch.device("cuda")
    d = bootstrap.subset(bootstrap.load_split("select", CELLS, CAND3D, CAND2D, DETS_MASTER), np.arange(N_FRAMES))
    for name in MODELS:
        cfg = tf.parse(name)
        items = d[tf.kind(cfg)] if tf.kind(cfg) != "base" else d["items"]
        for tag, steps in (("", None), ("_steps", EPOCHS * len(items))):  # --steps equal to the epochs must change nothing
            net, log = train_fusion.fit(items, EPOCHS, SEED, device, name, net_kwargs=cfg["net"], far_weight=2.0,
                                        n_extra=tf.n_extra(cfg), pairing=tf.pairing(cfg, d["top2d"]), steps=steps)
            torch.save(dict(model=net.state_dict(), settings=cfg), pathlib.Path(out) / f"master_{name}{tag}.pth")
    # a cut-off run: 1.5 epochs must stop after exactly that many updates
    cfg = tf.parse("R0+T2+NC")
    n = int(1.5 * N_FRAMES)
    counter = {"n": 0}
    orig = torch.optim.AdamW.step

    def counted(self, *a, **k):
        counter["n"] += 1
        return orig(self, *a, **k)
    torch.optim.AdamW.step = counted
    train_fusion.fit(d["items"], 3, SEED, device, "cut", net_kwargs=cfg["net"], far_weight=2.0, pairing=tf.pairing(cfg, d["top2d"]), steps=n)
    torch.optim.AdamW.step = orig
    json.dump(dict(requested=n, updates=counter["n"]), open(pathlib.Path(out) / "master_cut.json", "w"))
    gate = blend.train_blend(fixed_logits(d["items"]), d["items"], SEED, device, EPOCHS * N_FRAMES)
    torch.save(dict(model=gate.state_dict()), pathlib.Path(out) / "master_blend.pth")


def compare(out, report):
    out = pathlib.Path(out)
    res = {}
    for name in MODELS + ("blend",):
        t = torch.load(out / f"thesis_{name}.pth", map_location="cpu")
        for tag in ("", "_steps") if name != "blend" else ("",):
            m = torch.load(out / f"master_{name}{tag}.pth", map_location="cpu")
            diff = max(float((t["model"][k] - m["model"][k]).abs().max()) for k in t["model"])
            same_settings = t.get("settings") == m.get("settings")
            res[name + tag] = dict(max_abs_weight_diff=diff, identical=diff == 0.0, settings_equal=same_settings)
    res["cut_off"] = json.load(open(out / "master_cut.json"))
    res["setup"] = dict(frames=N_FRAMES, epochs=EPOCHS, seed=SEED, split="select")
    ok = all(v["identical"] for k, v in res.items() if isinstance(v, dict) and "identical" in v) and \
        all(v["settings_equal"] for k, v in res.items() if isinstance(v, dict) and "settings_equal" in v and k != "blend") and \
        res["cut_off"]["requested"] == res["cut_off"]["updates"]
    res["all_identical"] = ok
    json.dump(res, open(report, "w"), indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    {"thesis": lambda: thesis(sys.argv[2]), "master": lambda: master(sys.argv[2]),
     "compare": lambda: compare(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
