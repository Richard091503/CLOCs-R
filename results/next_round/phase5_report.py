"""Phase 5 report: the ceiling of a radar detector (results/next_round/PLAN.md, Phase 5).

    python results/next_round/phase5_report.py > results/next_round/phase5/report.md
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
P1, P2, P3, P4, P5 = (HERE / f"phase{k}" for k in (1, 2, 3, 4, 5))
ODDS = ("overall", "highway", "urban", "night", "rain")
GROUPS = ("overall", "highway", "urban", "night", "rain", "clear", "adverse")
BINS = ("all", "0-30", "30-50", "50-100", "100-200", "50-200")
MODELS = ("SECOND", "switch(R2+T2)", "R0+T2", "R2+T2", "R2+T2+K40", "R0+T2+K40+G", "R2+T2+K40+G")
FINAL = "R2+T2+K40+G"
THRESHOLD = 0.010
STORED = {"R0+T2": [P1 / "final_s123.json", P1 / "final_s456.json"], "R2+T2": [P1 / "final_s123.json", P1 / "final_s456.json"],
          "switch(R2+T2)": [P2 / "switch.json"], "R2+T2+K40": [P3 / "final_s123.json", P3 / "final_s456.json"],
          "R0+T2+K40+G": [P4 / "final_r0_s123.json", P4 / "final_s456.json"], FINAL: [P4 / "final_s123.json", P4 / "final_s456.json"]}


def stored(name):
    out = []
    for p in STORED[name]:
        if p.exists():
            out += json.load(open(p)).get(name, {}).get("runs", [])
    return sorted(out, key=lambda r: r["seed"])


def arr(runs, split, variant, g, b, metric="APCD"):
    return np.array([r[split][variant][g][b][metric] if g in r[split][variant] else np.nan for r in runs])


def gain(runs, split, variant, g, b):
    d = arr(runs, split, variant, g, b) - arr(runs, split, "none", g, b)
    if len(d) == 1:
        return f"{d[0]:+.4f}"
    return f"{d.mean():+.4f} ({d.min():+.4f} to {d.max():+.4f}, {int((d > 0).sum())} of {len(d)})"


def main():
    res = json.load(open(P5 / "ceiling.json"))
    splits = [s for s in ("select", "val") if s in res["counts"]]

    print("## Target cars: beyond 50 m, scored, no LiDAR point in the box, at least one radar return in the gate\n")
    for split in splits:
        print(f"### {split}\n")
        print("| Condition | Distance | Scored cars | No LiDAR point | Radar return (any) | Targets: no LiDAR + radar | "
              "Targets, % of cars | Targets SECOND already finds | No-LiDAR cars SECOND finds |")
        print("|" + "---|" * 9)
        for odd in ODDS:
            for b in ("50-100", "100-200", "50-200"):
                c = res["counts"][split].get(odd, {}).get(b)
                if not c:
                    continue
                print(f"| {odd} | {b} | {c['cars']:,} | {c['no_lidar']:,} ({100 * c['no_lidar'] / max(c['cars'], 1):.1f}%) | "
                      f"{c['radar']:,} ({100 * c['radar'] / max(c['cars'], 1):.1f}%) | {c['targets']:,} | "
                      f"{100 * c['targets'] / max(c['cars'], 1):.1f}% | {c['found']:,} | {c['no_lidar_found']:,} |")
        print()

    print("## Reproduction: \"none\" against the stored results (overall APCD, largest absolute difference over seeds)\n")
    print("| Model | Split | 50-200 | 100-200 | all |")
    print("|---|---|---|---|---|")
    sec = json.load(open(P1 / "detectors_fit/aimotive_second.json"))
    for name in MODELS:
        if name not in res:
            continue
        for split in splits:
            if name == "SECOND":
                old = {b: np.array([sec[split]["overall"][b]["APCD"]]) for b in ("50-200", "100-200", "all")}
            else:
                st = {r["seed"]: r for r in stored(name)}
                seeds = [r["seed"] for r in res[name]["runs"]]
                old = {b: np.array([st[s][split]["overall"][b]["APCD"] if s in st else np.nan for s in seeds]) for b in ("50-200", "100-200", "all")}
            new = {b: arr(res[name]["runs"], split, "none", "overall", b) for b in old}
            print(f"| {name} | {split} | " + " | ".join(f"{np.nanmax(np.abs(new[b] - old[b])):.1e}" for b in old) + " |")
    print()

    for split in splits:
        print(f"## {split}: APCD beyond 50 m with and without the injected cars (mean of 6 seeds; ceiling seed against seed)\n")
        print("| Model | 50-200 none | low | perfect | 100-200 none | low | perfect | Low ceiling, 50-200 | Low ceiling, 100-200 | "
              "Perfect ceiling, 50-200 | Perfect ceiling, 100-200 | Injected kept by NMS, low |")
        print("|" + "---|" * 12)
        n_inj = sum(res["counts"][split]["overall"][b]["targets"] for b in ("50-100", "100-200"))
        for name in MODELS:
            if name not in res:
                continue
            rs = res[name]["runs"]
            cells = [f"{arr(rs, split, v, 'overall', b).mean():.4f}" for b in ("50-200", "100-200") for v in ("none", "low", "perfect")]
            kept = np.mean([r[split]["low"]["injected_kept"] for r in rs])
            print(f"| {name} | " + " | ".join(cells) + " | " +
                  " | ".join(gain(rs, split, v, "overall", b) for v in ("low", "perfect") for b in ("50-200", "100-200")) +
                  f" | {kept:.0f} of {n_inj} |")
        print()

    for name in (FINAL, "SECOND"):
        if name not in res:
            continue
        for split in splits:
            rs = res[name]["runs"]
            print(f"### {split}, {name}: every condition and bin\n")
            print("| Condition | Distance | none | low | perfect | Low ceiling | Perfect ceiling |")
            print("|---|---|---|---|---|---|---|")
            for g in GROUPS:
                for b in BINS:
                    a = arr(rs, split, "none", g, b)
                    if np.isnan(a).all():
                        continue
                    print(f"| {g} | {b} | {a.mean():.3f} | {arr(rs, split, 'low', g, b).mean():.3f} | "
                          f"{arr(rs, split, 'perfect', g, b).mean():.3f} | {gain(rs, split, 'low', g, b)} | {gain(rs, split, 'perfect', g, b)} |")
            print()

    for split in splits:
        print(f"### {split}: low ceiling at 50-200 m by condition, every model (mean)\n")
        print("| Model | " + " | ".join(ODDS) + " |")
        print("|" + "---|" * (1 + len(ODDS)))
        for name in MODELS:
            if name not in res:
                continue
            rs = res[name]["runs"]
            print(f"| {name} | " + " | ".join(
                f"{(arr(rs, split, 'low', g, '50-200') - arr(rs, split, 'none', g, '50-200')).mean():+.4f}" for g in ODDS) + " |")
        print()

    if FINAL in res and "select" in splits:
        d = arr(res[FINAL]["runs"], "select", "low", "overall", "50-200") - arr(res[FINAL]["runs"], "select", "none", "overall", "50-200")
        verdict = "worth building" if d.mean() >= THRESHOLD else "not worth building"
        print(f"**Rule:** the low ceiling of {FINAL} at 50-200 m on select is {d.mean():+.4f} against a threshold of "
              f"+{THRESHOLD:.3f}, so by the pre-registered rule a radar detector is **{verdict}**.")


if __name__ == "__main__":
    main()
