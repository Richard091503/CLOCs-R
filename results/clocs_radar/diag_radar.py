"""Checks on the radar data before any features are used: Eq. 1 on every sequence, and agreement with labeled velocities.

Run from the repo root: python results/clocs_radar/diag_radar.py
"""
import json
import pathlib

import numpy as np

from clocs_r import aimotive, splits

OUT = pathlib.Path(__file__).parent / "diag_radar.json"
BINS = [(0, 30), (30, 50), (50, 100), (100, 200)]
SWEEPS, DELTA0, KAPPA = 3, 0.5, 0.02  # the starting gate, before selection


def in_gate(pts, box, delta):
    c, s = np.cos(box[6]), np.sin(box[6])
    dx, dy = pts[:, 0] - box[0], pts[:, 1] - box[1]
    return (np.abs(dx * c + dy * s) <= box[3] / 2 + delta) & (np.abs(-dx * s + dy * c) <= box[4] / 2 + delta)


def check_sequence(seq):
    """Returns far from every labeled object should be static (near 0 after Eq. 1), and returns on a car should match its label."""
    off, agree = [], []
    for f in seq.kept[::2]:
        pts, vel = seq.radar_sweep(f)
        if not len(pts):
            continue
        near = np.zeros(len(pts), bool)
        for b in seq.boxes(f)[0]:
            near |= in_gate(pts, b, 2 + 0.02 * np.hypot(b[0], b[1]))
        off.append(vel[~near])
        ego = seq.ego_velocity(f)[:2]
        for o in seq.meta["labels"].get(f, []):
            if o["ObjectType"] not in splits.CAR_CLASSES:
                continue
            b = seq.boxes(f)[0][seq.meta["labels"][f].index(o)]
            rho = np.hypot(b[0], b[1])
            q = in_gate(pts, b, DELTA0 + KAPPA * rho)
            if q.any() and rho > 1:
                rel = np.array([o["Relative Velocity X"], o["Relative Velocity Y"]], dtype=float)
                agree.append(np.median(vel[q]) - (rel + ego) @ (b[:2] / rho))
    off = np.concatenate(off) if off else np.zeros(0)
    agree = np.abs(agree)
    return dict(seq=seq.name, odd=seq.odd, off_label_returns=len(off), off_label_near_zero=float(np.mean(np.abs(off) < 1)) if len(off) else None,
                off_label_median=float(np.median(off)) if len(off) else None, cars=len(agree),
                label_agreement=float(np.median(agree)) if len(agree) else None, label_within_2=float(np.mean(agree < 2)) if len(agree) else None)


def check_cars(split, stride=5):
    """Labeled cars in the scored region: do they have returns in the gate, by distance?"""
    out = {f"{lo}-{hi}": {m: [] for m in (1, 3)} for lo, hi in BINS}
    for seq, frame in splits.frames(split)[::stride]:
        boxes = seq.boxes(frame, splits.CAR_CLASSES)[0]
        if len(boxes):
            boxes = boxes[aimotive.in_region(boxes) & aimotive.in_camera_view(seq, boxes)]
        if not len(boxes):
            continue
        rad = {m: seq.radar_accumulated(frame, m)[0] for m in out["0-30"]}
        for b in boxes:
            rho = np.hypot(b[0], b[1])
            name = next((f"{lo}-{hi}" for lo, hi in BINS if lo <= rho < hi), None)
            if name:
                for m, pts in rad.items():
                    out[name][m].append(int(in_gate(pts, b, DELTA0 + KAPPA * rho).sum()))
    return {n: {f"M{m}": dict(cars=len(v), with_return=float(np.mean(np.array(v) > 0)), mean_returns=float(np.mean(v))) for m, v in d.items()} for n, d in out.items()}


def main():
    seqs, cars = {}, {}
    for split in ("train", "holdout", "val"):
        seqs[split] = [check_sequence(aimotive.Sequence(p)) for p in splits.scenes(split)]
        s = [x for x in seqs[split] if x["label_agreement"] is not None]
        near = [x["off_label_near_zero"] for x in seqs[split] if x["off_label_near_zero"] is not None]
        print(split, len(seqs[split]), "seqs; off-label near zero: median", round(np.median(near), 3), "min", round(min(near), 3),
              "; label agreement m/s: median", round(np.median([x["label_agreement"] for x in s]), 3), "worst", round(max(x["label_agreement"] for x in s), 2))
        for x in sorted(s, key=lambda x: -x["label_agreement"])[:4]:
            print("   worst agreement", x["seq"], x["odd"], {k: x[k] if isinstance(x[k], int) else round(x[k], 2) for k in ("cars", "label_agreement", "label_within_2", "off_label_near_zero", "off_label_returns")})
        for x in sorted(seqs[split], key=lambda x: x["off_label_near_zero"] if x["off_label_near_zero"] is not None else 1)[:4]:
            print("   lowest static", x["seq"], x["odd"], {k: x[k] if isinstance(x[k], int) else round(x[k], 2) for k in ("cars", "label_agreement", "label_within_2", "off_label_near_zero", "off_label_returns")})
    for split in ("holdout", "val"):
        cars[split] = check_cars(split)
        print(split, json.dumps({k: {m: round(v[m]["with_return"], 2) for m in v} for k, v in cars[split].items()}))
    with open(OUT, "w") as f:
        json.dump(dict(sequences=seqs, cars=cars, gate=dict(delta0=DELTA0, kappa=KAPPA)), f, indent=1)


if __name__ == "__main__":
    main()
