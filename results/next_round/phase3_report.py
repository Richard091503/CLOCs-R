"""Phase 3 report: Stage A and B on select by the rule of PLAN.md, then val once for the configuration it picked.

    python results/next_round/phase3_report.py > results/next_round/phase3/report.md
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import phase3_decide as rule  # noqa: E402

P1, P2, P3 = HERE / "phase1", HERE / "phase2", HERE / "phase3"
GROUPS = ("overall", "highway", "urban", "night", "rain", "clear", "adverse")
BINS = ("all", "0-30", "30-50", "50-100", "100-200", "50-200")
METRICS = ("APCD", "APBEV@0.3", "AP3D@0.5", "recall@BEV0.3")


def load(paths, name):
    out = []
    for p in paths:
        if p.exists():
            out += json.load(open(p)).get(name, {}).get("runs", [])
    return sorted(out, key=lambda r: r["seed"])


def arr(rs, split, g, b, metric="APCD"):
    return np.array([r[split][g][b][metric] for r in rs])


def ms(a):
    return f"{a.mean():.4f} ± {a.std(ddof=1):.4f}" if len(a) > 1 else f"{a.mean():.4f}"


def stage_table(names):
    print("| Configuration | Params | 0-30 | 30-50 | 50-100 | 100-200 | 50-200 (mean ± SD) | Δ 50-200 | Δ 100-200 | Beats reference | Beats switch |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        try:
            rs = rule.runs(n)
        except KeyError:
            continue
        a = {b: arr(rs, "select", "overall", b) for b in BINS}
        ok_ref = rule.beats(n, rule.REF)[0] if n != rule.REF else None
        ok_sw = rule.beats(n, "switch(R2+T2)")[0]
        ref = {b: arr(rule.runs(rule.REF), "select", "overall", b).mean() for b in ("50-200", "100-200")}
        print(f"| {n} | {rs[0].get('params', '')} | " + " | ".join(f"{a[b].mean():.3f}" for b in ("0-30", "30-50", "50-100", "100-200")) +
              f" | {ms(a['50-200'])} | {a['50-200'].mean() - ref['50-200']:+.4f} | {a['100-200'].mean() - ref['100-200']:+.4f} | "
              f"{'-' if ok_ref is None else 'yes' if ok_ref else 'no'} | {'yes' if ok_sw else 'no'} |")
    print()


def main():
    print("## Stage A: each change alone on R2+T2 (select, seeds 1-3, APCD)\n")
    stage_table([rule.REF] + [f"{rule.REF}+{t}" for t in rule.TOKENS])
    p1 = json.load(open(P1 / "select_insample.json"))["R2+T2"]["runs"]
    print("Reproduction check, R2+T2 on select beyond 50 m: Phase 1", [round(r["select"]["overall"]["50-200"]["APCD"], 4) for r in p1],
          "this phase", [round(r["select"]["overall"]["50-200"]["APCD"], 4) for r in rule.runs(rule.REF)], "\n")
    dec = P3 / "decision_A.json"
    if not dec.exists():
        return
    kept = json.load(open(dec))["kept"]
    print(f"Stage A kept: {', '.join(kept) if kept else 'nothing'}.\n")
    stage_b = json.load(open(P3 / "stageB.json")) if (P3 / "stageB.json").exists() else {}
    if stage_b:
        print("## Stage B (select, seeds 1-3)\n")
        stage_table(list(stage_b))
    final = (P3 / "final_config.txt").read_text().strip() if (P3 / "final_config.txt").exists() else None
    if final is None:
        return
    print(f"Final configuration: **{final}**.\n")
    if final == rule.REF:
        print("Nothing was adopted, so val is not scored again (Phase 1 has R2+T2 on val).")
        return
    r0 = final.replace("R2+T2", "R0+T2")
    sec = json.load(open(P1 / "detectors_fit/aimotive_second.json"))
    f1 = [P1 / "final_s123.json", P1 / "final_s456.json"]
    models = {"R0+T2": load(f1, "R0+T2"), "R2+T2": load(f1, "R2+T2"),
              "switch(R0+T2)": load([P2 / "switch.json"], "switch(R0+T2)"), "switch(R2+T2)": load([P2 / "switch.json"], "switch(R2+T2)"),
              r0: load([P3 / "final_r0_s123.json", P3 / "final_s456.json"], r0),
              final: load([P3 / "final_s123.json", P3 / "final_s456.json"], final)}
    names = list(models)
    for split in ("select", "val"):
        print(f"## {split}: APCD by condition and distance (mean of 6 seeds; radar gain of the final model seed against seed)\n")
        print("| Condition | Distance | Cars | SECOND | " + " | ".join(names) + " | final − SECOND | final − switch(R2+T2) | radar gain |")
        print("|" + "---|" * (7 + len(names)))
        for g in GROUPS:
            for b in BINS:
                e = sec[split].get(g, {}).get(b)
                if not isinstance(e, dict) or not e.get("n_gt"):
                    continue
                fa, sw = arr(models[final], split, g, b), arr(models["switch(R2+T2)"], split, g, b)
                d = fa - arr(models[r0], split, g, b)
                print(f"| {g} | {b} | {e['n_gt']:,} | {e['APCD']:.3f} | " + " | ".join(f"{arr(models[m], split, g, b).mean():.3f}" for m in names) +
                      f" | {fa.mean() - e['APCD']:+.3f} | {fa.mean() - sw.mean():+.3f} | "
                      f"{d.mean():+.3f} ({d.min():+.3f} to {d.max():+.3f}, {int((d > 0).sum())} of {len(d)}) |")
        print(f"\n### {split}: other metrics, overall beyond 50 m\n")
        print("| Metric | SECOND | " + " | ".join(names) + " |")
        print("|" + "---|" * (2 + len(names)))
        for metric in METRICS:
            print(f"| {metric} | {sec[split]['overall']['50-200'][metric]:.3f} | " +
                  " | ".join(f"{arr(models[m], split, 'overall', '50-200', metric).mean():.3f}" for m in names) + " |")
        print()


if __name__ == "__main__":
    main()
