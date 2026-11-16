"""Tables for ablation/README.md from the outputs of clocs_r/ablation_full.py.

    python results/full_dataset/ablation/make_tables.py
Writes ablation/tables.md. Holdout numbers are the selection metric (center-distance AP beyond 50 m, mean of seeds 1-3);
val numbers come from val.json (seeds 1-3) and, for the two final configurations, extra_seeds.json (seeds 4-6).
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
DETECTORS = HERE.parent / "detectors" / "aimotive_second.json"
FEATURES = {"R0": "none", "R1": "r", "R2": "v", "R3": "r, v", "R4": "r, v, e", "R5": "r, Δ, o", "R6": "r, v, Δ", "R7": "r, v, e, Δ, o"}
CHANGES = {"T1": "far targets", "T2": "residual on SECOND's score", "T3": "logit input", "T4a": "5 sweeps, motion-corrected",
           "T4b": "10 sweeps, motion-corrected", "T5": "RCS channel"}


def load(name):
    p = HERE / name
    return json.load(open(p)) if p.exists() else {}


def vals(entry, split, group="overall", bin_="50-200", metric="APCD"):
    return [r[split][group][bin_][metric] for r in entry["runs"] if split in r and group in r[split]]


def fmt(v, d=3, signed=False):
    """Mean (min to max), with a real minus sign; signed adds + to positive numbers (for gains)."""
    if not v:
        return "n/a"
    f = (lambda x: f"{x:+.{d}f}" if signed else f"{x:.{d}f}")
    g = lambda x: f(x).replace("-", "−")
    return f"{g(np.mean(v))} ({g(min(v))} to {g(max(v))})" if len(v) > 1 else g(v[0])


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main():
    dec = load("decisions.json")
    second = json.load(open(DETECTORS))
    sec_h, sec_v = second["holdout"]["overall"]["50-200"]["APCD"], second["val"]["overall"]["50-200"]["APCD"]
    val, extra, nocam = load("val.json"), load("extra_seeds.json"), load("val_nocam.json")
    final, twin = dec["final"]["clocs_r"], dec["final"]["clocs"]
    md = []

    g = dec["gate"]
    rows = []
    for tag, v in sorted(g["holdout"].items(), key=lambda kv: -kv[1]["mean"]):
        d0, k = tag[1:].split("_k")
        sel = [float(d0), float(k)] == g["selected"]
        rows.append([("**" if sel else "") + f"δ0 = {d0} m, κ = {k}" + ("**" if sel else ""), f"{v['mean']:.4f}",
                     ", ".join(f"{x:.4f}" for x in v["seeds"]), "selected" if sel else ""])
    md.append(f"## 1. Gate (R3, holdout)\n\nCenter-distance AP beyond 50 m, mean of seeds 1-3. SECOND alone: {sec_h:.4f}.\n\n"
              + table(["Gate", "Mean", "Seeds", ""], rows))

    fs = dec["feature_set"]
    rows = []
    for n in FEATURES:
        v = fs["holdout"][n]
        rows.append([f"**{n}**" if n == fs["selected"] else n, FEATURES[n], f"{v['mean']:.4f}", ", ".join(f"{x:.4f}" for x in v["seeds"]),
                     fmt(vals(val[n], "val")) if n in val else "n/a", "selected" if n == fs["selected"] else ("reference" if n == "R0" else "")])
    md.append(f"\n## 2. Feature sets (gate δ0 = {g['selected'][0]} m, κ = {g['selected'][1]})\n\n"
              f"Holdout chose; val is shown for context and played no part. SECOND alone: holdout {sec_h:.4f}, val {sec_v:.4f}. "
              "r = number of radar returns in the gate (log), v = Doppler speed, e = Doppler spread, Δ = range residual, o = offset.\n\n"
              + table(["Set", "Features", "Holdout mean", "Holdout seeds", "Val (seeds 1-3)", ""], rows))

    ch = dec["changes"]
    S = ch["base"]
    rows = [[S, "none (base)", f"{fs['holdout'][S]['mean']:.4f}", "", fmt(vals(val[S], "val")) if S in val else "n/a", "base"]]
    for n, v in ch["holdout"].items():
        parts = [p for p in n.split("+")[1:]]
        what = " + ".join(CHANGES[p] for p in parts)
        adopted = len(parts) == 1 and parts[0] in ch["adopted"]
        role = "combination" if len(parts) > 1 else ("adopted" if adopted else "not adopted")
        if n == ch["final"]:
            role += ", **final**"
        rows.append([f"**{n}**" if n == ch["final"] else n, what, f"{v['mean']:.4f}", ", ".join(f"{x:.4f}" for x in v["seeds"]),
                     fmt(vals(val[n], "val")) if n in val else "n/a", role])
    md.append(f"\n## 3. Fusion changes on {S}\n\nA change is adopted if its holdout mean exceeds {S}'s by more than the standard deviation "
              f"of {S}'s seeds, that is above **{ch['threshold']:.4f}**. Adopted: {', '.join(ch['adopted']) or 'none'}."
              + (f" Combination tested: {ch['combination']}." if ch["combination"] else "") + "\n\n"
              + table(["Configuration", "Change", "Holdout mean", "Holdout seeds", "Val (seeds 1-3)", "Role"], rows))

    def all_seeds(n):
        runs = list(val[n]["runs"]) + [r for r in extra.get(n, {}).get("runs", []) if r["seed"] not in {x["seed"] for x in val[n]["runs"]}]
        return dict(val[n], runs=runs)

    F, T = all_seeds(final), all_seeds(twin)
    rows = []
    for gname, label in (("overall", "All conditions"), ("clear", "Clear"), ("highway", "Highway"), ("urban", "Urban"),
                         ("adverse", "Adverse"), ("night", "Night"), ("rain", "Rain")):
        f, t = vals(F, "val", gname), vals(T, "val", gname)
        s = second["val"][gname]["50-200"]["APCD"]
        rows.append([label, f'{second["val"][gname]["50-200"]["n_gt"]:,}', f"{s:.3f}", fmt(t), fmt(f), fmt([a - b for a, b in zip(f, t)], signed=True),
                     f"{np.mean(f) - s:+.3f}".replace("-", "−")])
    md.append(f"\n## 4. Final configurations on val, cars beyond 50 m (center-distance AP, {len(F['runs'])} seeds)\n\n"
              f"Final CLOCs-R: **{final}**. Final CLOCs: **{twin}**. Radar gain is computed seed against seed.\n\n"
              + table(["Condition", "Cars", "SECOND", f"CLOCs ({twin})", f"CLOCs-R ({final})", "Radar gain", "CLOCs-R vs SECOND"], rows))

    rows = []
    for b in ("all", "0-30", "30-50", "50-100", "100-200", "50-200"):
        f, t = vals(F, "val", "overall", b), vals(T, "val", "overall", b)
        rows.append([b + (" m" if b != "all" else ""), f'{second["val"]["overall"][b]["n_gt"]:,}', f"{second['val']['overall'][b]['APCD']:.3f}",
                     fmt(t), fmt(f), fmt([a - c for a, c in zip(f, t)], signed=True)])
    md.append("\n### By distance, all conditions\n\n" + table(["Distance", "Cars", "SECOND", twin, final, "Radar gain"], rows))

    if nocam:
        rows = []
        for n, label in ((twin, f"CLOCs ({twin})"), (final, f"CLOCs-R ({final})")):
            for split in ("holdout", "val"):
                with_cam = vals(all_seeds(n), split) if split == "val" else vals(val[n], split)
                rows.append([label, split, fmt(with_cam), fmt(vals(nocam[n], split)),
                             fmt(vals(all_seeds(n) if split == "val" else val[n], split, "overall", "100-200")), fmt(vals(nocam[n], split, "overall", "100-200"))])
        md.append(f"\n## Camera removed at test time\n\nCenter-distance AP. SECOND alone: holdout {sec_h:.3f}, val {sec_v:.3f}. "
                  "The network rarely sees candidates without any 2D box during training, so these inputs are unusual for it.\n\n"
                  + table(["Model", "Split", "With camera, beyond 50 m", "Camera removed, beyond 50 m", "With camera, 100-200 m",
                           "Camera removed, 100-200 m"], rows))

    lat = load("latency.json")
    if lat:
        rows = [[n, f"{v['features_median_ms']:.1f} ({v['features_p95_ms']:.1f})" if n != "R0" else "none", f"{v['fusion_median_ms']:.2f}",
                 f"{v['features_median_ms'] + v['fusion_median_ms']:.1f}", f"{v['params']:,}"] for n, v in lat["sets"].items()]
        md.append(f"\n## Latency per frame (median of {lat['frames']} val frames)\n\nRadar features on one CPU thread in Python; fusion on the GPU; "
                  "detectors excluded. The fusion forward times are dominated by GPU launch overhead and vary between runs (R0 "
                  "was timed first and includes warm-up), so only the radar-feature times separate the sets.\n\n" + table(["Set", "Radar features, ms (95th pct.)", "Fusion forward, ms", "Total, ms", "Parameters"], rows))
    (HERE / "tables.md").write_text("\n\n".join(md) + "\n")
    print("wrote", HERE / "tables.md")


if __name__ == "__main__":
    main()
