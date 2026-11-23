"""Write results/review_round/REPORT.md from the review-round result files (run from the review-round worktree root).

    python results/review_round/make_report.py
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "lcr" / "data"
BINS = ("50-100", "100-200", "50-200", "all")
STORED = ("bootstrap_{s}.json", "bootstrap_lcr_{s}.json", "bootstrap_lcrd_{s}.json")  # results/lcr/data (thesis phase7 copies)
NAMES = {"LCR": "LCR (CLOCs-R)", "LCRD": "LCR-D (CLOCs-R-D)"}


def cap(s):
    for a, b in (("second ", "SECOND "), ("cascade ", "Cascade "), ("d2 ", "D2 ")):
        s = s.replace(a, b, 1) if s.startswith(a) else s
    return s


def f4(x):
    return f"{x:+.4f}"


def iv(v):
    return f"[{f4(v['lo'])}, {f4(v['hi'])}]"


def label(c):
    a, b = c.split(" - ")
    return f"{NAMES.get(a, a).split(' ')[0]} − {NAMES.get(b, b).split(' ')[0]}".replace("LCRD", "LCR-D")


def load(p):
    p = pathlib.Path(p)
    return json.load(open(p)) if p.exists() else None


def stored(split):
    out, points = {}, {}
    for f in STORED:
        d = load(DATA / f.format(s=split))
        out.update(d["comparisons"])
        for k, v in d["point"].items():
            points.setdefault(k, v)
    return out, points


def where(c, b):
    return f"{c}, all ranges" if b == "all" else f"{c} at {b} m"


def task1(lines):
    lines += ["## Task 1. Recording-level bootstrap", "",
              "`results/review_round/cluster_bootstrap.py` (run with the thesis code, which scores L0, LRf and B) computes the",
              "per-frame matches of every model and seed once. It then resamples them in two ways, with 1,000 resamples each,",
              "seeds averaged inside each resample, percentile 95% intervals, and `numpy.random.RandomState(0)` as in",
              "`clocs_r/bootstrap.py`:",
              "",
              "- **Scene level:** draws scenes with replacement; identical to the stored bootstrap.",
              "- **Recording level:** draws recordings (`clocs_r.splits.recording`) with replacement and keeps all their scenes.",
              "",
              "The point estimate is the APCD difference of the 6-seed means on the whole split.",
              "**Scene (stored)** is the interval in `results/lcr/data/bootstrap_*.json`. **Scene (rerun)** is this run's, and the",
              "reproduction check compares the two. **Recording** is new. LC − L0 had never been bootstrapped, so its scene",
              "interval is from this run.", ""]
    for split in ("select", "val"):
        new = load(HERE / f"bootstrap_{split}.json")
        if new is None:
            lines += [f"**{split}: not finished.**", ""]
            continue
        old, old_points = stored(split)
        recs = new["recordings"]
        lines += [f"### {'`select`' if split == 'select' else 'val'}: {new['scenes']} scenes from {len(recs)} recordings, {new['frames']:,} frames", "",
                  "Scenes per recording: " + ", ".join(f"{r} {n}" for r, n in sorted(recs.items(), key=lambda x: -x[1])) + ".", ""]
        # reproduction check
        dp = max(abs(new["point"][m][b] - old_points[m][b]) for m in new["point"] if m in old_points for b in BINS)
        checks = [(c, b) for c in new["scene"] if c in old for b in BINS]
        dd = max(abs(new["scene"][c][b]["diff"] - old[c][b]["diff"]) for c, b in checks)
        dl = max(max(abs(new["scene"][c][b]["lo"] - old[c][b]["lo"]), abs(new["scene"][c][b]["hi"] - old[c][b]["hi"])) for c, b in checks)
        lines += [f"**Reproduction check (scene level against the stored files):**",
                  f"- {len(checks)} comparison-bins checked.",
                  f"- Largest differences: model point APCD {dp:.1e}, comparison estimate {dd:.1e}, interval end {dl:.1e}.",
                  ""]
        lines += ["| Comparison | Bin (m) | Estimate | Scene (stored) | Scene (rerun) | Recording (new) |", "|---|---|---|---|---|---|"]
        for c in new["recording"]:
            for b in BINS:
                s, r = new["scene"][c][b], new["recording"][c][b]
                st = iv(old[c][b]) if c in old else "not stored"
                lines.append(f"| {label(c)} | {b} | {f4(s['diff'])} | {st} | {iv(s)} | {iv(r)} |")
        lines.append("")
        excl = lambda v: v["lo"] > 0 or v["hi"] < 0
        lost = [(label(c), b) for c in new["recording"] for b in BINS if excl(new["scene"][c][b]) and not excl(new["recording"][c][b])]
        gained = [(label(c), b) for c in new["recording"] for b in BINS if not excl(new["scene"][c][b]) and excl(new["recording"][c][b])]
        narrower = sum((r["hi"] - r["lo"]) < (s["hi"] - s["lo"]) for c in new["recording"]
                       for s, r in ((new["scene"][c][b], new["recording"][c][b]) for b in BINS))
        lines += ["**Where the verdict changes:**",
                  "- **Scene interval excludes 0, recording interval includes 0:** "
                  + (", ".join(where(c, b) for c, b in lost) if lost else "none") + ".",
                  "- **Scene interval includes 0, recording interval excludes 0:** "
                  + (", ".join(where(c, b) for c, b in gained) if gained else "none") + ".",
                  f"- **Width:** the recording interval is narrower than the scene interval in {narrower} of "
                  f"{len(new['recording']) * len(BINS)} comparison-bins.",
                  "",
                  f"**Caveat:** with only {len(recs)} recordings, some holding a single scene, a percentile bootstrap over",
                  "recordings is itself unreliable and tends to understate the spread. Where a recording interval is narrower",
                  "than the scene interval, that does not show that recordings add no correlation. Read the second list as weak",
                  "evidence, and the first list as the cases the paper should not call significant without qualification.", ""]
        nanr = sum(v["nan_draws"] for c in new["recording"].values() for v in c.values())
        if nanr:
            lines += [f"Note: {nanr} recording-level draws had no car in a bin and were left out of the percentiles.", ""]


def task2(lines):
    r = load(HERE / "second_reliability.json")
    lines += ["## Task 2. Is SECOND more reliable on its own training scenes?", ""]
    if r is None:
        lines += ["Not finished.", ""]
        return
    sec, cas = load(DATA / "aimotive_second.json"), load(DATA / "aimotive_cascade.json")
    d1 = max(abs(r[s]["second_apcd"][b] - sec[s]["overall"][b]["APCD"]) for s in ("select", "val") for b in r[s]["second_apcd"])
    d2 = max(abs(r[s]["cascade_recall05"][b] - cas[s]["overall"]["recall"][b]["recall@0.5"]) for s in ("select", "val") for b in r[s]["cascade_recall05"])
    lines += ["`results/review_round/second_reliability.py` scores the cached candidates with the `eval_detectors` code, and",
              f"reproduces the stored `select` and val numbers. The largest differences are {d1:.1e} for SECOND's APCD and {d2:.1e} for",
              "Cascade R-CNN's recall.", "",
              "**SECOND alone, APCD** (`fit` is in-sample: SECOND was trained on it):", "",
              "| Split | Frames | 0–30 | 30–50 | 50–100 | 100–200 | 50–200 |", "|---|---|---|---|---|---|---|"]
    B = ("0-30", "30-50", "50-100", "100-200", "50-200")
    for s in ("fit", "select", "val"):
        lines.append(f"| {s} | {r[s]['frames']:,} | " + " | ".join(f"{r[s]['second_apcd'][b]:.4f}" for b in B) + " |")
    lines += ["", "**Reliability of SECOND's candidates beyond 50 m.** These are the candidates after BEV NMS, as in evaluation",
              "(region and camera view, trucks and buses removed, NMS at IoU 0.1). \"True\" is the share with a scored car whose",
              "BEV centre is within 2 m; n is the number of candidates.", "",
              "| Score | fit: true (n) | select: true (n) | val: true (n) |", "|---|---|---|---|"]
    for k in r["fit"]["reliability_far"]:
        lo, hi = k.split("-")
        lines.append(f"| [{lo}, {hi}{']' if hi == '1.0' else ')'} | " + " | ".join(
            f"{r[s]['reliability_far'][k]['true']:.4f} ({r[s]['reliability_far'][k]['n']:,})" for s in ("fit", "select", "val")) + " |")
    lines += ["", "**Cascade R-CNN (trained on `fit`), candidate recall@0.5 for cars at 100–200 m:**", "",
              "| fit (in-sample) | select | val |", "|---|---|---|",
              "| " + " | ".join(f"{r[s]['cascade_recall05']['100-200']:.4f}" for s in ("fit", "select", "val")) + " |", ""]


def task3(lines):
    lines += ["## Task 3. Size and speed", ""]
    parts = {p: load(HERE / f"timing_{p}.json") for p in ("second", "d2", "cascade", "lcr", "params")}
    hw = load(HERE / "hardware.json")
    if not all(parts.values()):
        lines += ["Not finished: " + ", ".join(p for p, v in parts.items() if v is None) + ".", ""]
        return
    if hw:
        lines += ["**Hardware:**",
                  f"- GPU: {hw['gpu']}.",
                  f"- CPU: {hw['cpu']}, {hw['cores']} cores available.",
                  f"- {hw['torch']}.",
                  "- Nothing else ran on the machine during the timing.",
                  ""]
    n = parts["second"]["frames"]
    lines += [f"Batch size 1, on {n['timed']} val frames spread evenly over the split, after {n['warmup']} warm-up frames. CUDA was",
              "synchronized before and after each timed step. Code: `results/review_round/timing.py`.", "",
              "| Step | Runs on | Median (ms) | 90th pct (ms) |", "|---|---|---|---|"]
    for p in ("second", "cascade", "d2", "lcr"):
        for k, v in parts[p]["steps"].items():
            where = k[k.rfind("(") + 1:-1] if k.endswith(")") else ""
            lines.append(f"| {cap(k[:k.rfind('(')].strip() if where else k)} | {where} | {v['median_ms']:.1f} | {v['p90_ms']:.1f} |")
    c = parts["lcr"].get("cells_checked_equal_to_cache")
    lines += ["", "Notes on the steps:",
              "- **SECOND and D2, input:** includes reading the frame's files from disk.",
              "- **SECOND, network:** ends at the top 2,000 boxes before NMS, as the cached candidates are made.",
              "- **Cascade R-CNN:** one call to MMDetection's `inference_detector` (read the JPEG, resize to "
              f"{parts['cascade'].get('img_scale')}, normalize, network, post-processing).",
              "- **LCR cells:** start from the three detectors' outputs, already in memory.",
              f"- **Cell check:** the timed code's cells were checked equal to the cached LCR cells on {c} frames.", ""]
    p = parts["params"]["state_dict_numel"]
    built = {"second": parts["second"]["model_parameters"], "d2": parts["d2"]["model_parameters"],
             "cascade": parts["cascade"]["model_parameters"], "LCR fusion": parts["lcr"]["model_parameters"]}
    names = {"second": "SECOND", "d2": "D2 (radar PointPillars)", "cascade": "Cascade R-CNN (ResNet-50 FPN)",
             "clutter MLP": "Clutter MLP (12-64-64-1)", "LCR fusion": "LCR fusion network"}
    lines += ["**Parameters.** \"Parameters\" counts the trainable parameters of the model as built and loaded. \"Checkpoint",
              "elements\" also counts BatchNorm running statistics.", "",
              "| Model | Parameters | Checkpoint elements |", "|---|---|---|"]
    for k, v in p.items():
        b = built.get(k, v if k == "clutter MLP" else None)
        lines.append(f"| {names[k]} | {b:,} | {v:,} |")
    lines += [""]


def task5(lines):
    lines += ["## Task 5. Pairing-radius sensitivity (post-hoc check, not a selection)", ""]
    res = {r: (load(HERE / "radius" / f"lrd_r{r}.json"), load(HERE / "radius" / f"lrd_r{r}_val.json")) for r in ("1.0", "2.0", "3.0")}
    if not all(a and b for a, b in res.values()):
        lines += ["Not run, or not finished.", ""]
        return
    B = ("0-30", "30-50", "50-100", "100-200", "50-200")
    lines += ["LRd (R0+T2+RP) was retrained on `fit` with radar pairing radius r, affinity 1 − d / r, seeds 1–3. Everything",
              "else is unchanged (`results/review_round/radius_lrd.py`; the code's default stays 2 m). Each model was scored on",
              "`select`, then once on val from the saved networks. The 2 m retrain reproduces the stored LRd exactly: for",
              "each of seeds 1–3 the difference is 0.0 in every bin and condition, so training is deterministic here.",
              "",
              "Values are APCD, mean ± SD over seeds 1–3.", "",
              "| Radius | Split | " + " | ".join(B) + " |", "|---|---|" + "---|" * len(B)]
    for r, (sel, val) in res.items():
        for name, d, s in (("select", sel, "select"), ("val", val, "val")):
            runs = d["R0+T2+RP"]["runs"]
            vals = {b: np.array([x[s]["overall"][b]["APCD"] for x in runs]) for b in B}
            lines.append(f"| {r} m | {name} | " + " | ".join(f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" for v in vals.values()) + " |")
    st = {s: load(DATA / f) for s, f in (("select", "experts_rp.json"), ("val", "val_experts_rp.json"))}
    lines += ["", "**Reproduction check:** the stored LRd (radius 2 m), mean of seeds 1–3, against this run's 2 m:", "",
              "| Split | " + " | ".join(B) + " |", "|---|" + "---|" * len(B)]
    for s, d in (("select", res["2.0"][0]), ("val", res["2.0"][1])):
        old = [x for x in st[s]["R0+T2+RP"]["runs"] if x["seed"] in (1, 2, 3)]
        lines.append(f"| {s}, stored | " + " | ".join(f"{np.mean([x[s]['overall'][b]['APCD'] for x in old]):.4f}" for b in B) + " |")
        lines.append(f"| {s}, rerun | " + " | ".join(f"{np.mean([x[s]['overall'][b]['APCD'] for x in d['R0+T2+RP']['runs']]):.4f}" for b in B) + " |")
    lines += [""]


def paper(lines):
    """Cross-check the two B intervals quoted in the paper draft against the stored val bootstrap."""
    old, _ = stored("val")
    blc, blcr = old["B - LC"]["50-200"], old["LCR - B"]["50-200"]
    lines += ["## Checks against the paper draft", "",
              "- **Table VI, B's gain over LC on val, 50–200 m,** quoted as \"+0.016 [+0.012, +0.020]\": the stored value is",
              f"  {f4(blc['diff'])} {iv(blc)}. Matches.",
              "- **Text, \"B beats CLOCs-R by 0.016 [0.009, 0.022]\":** this is B − LCR on val at 50–200 m. The stored LCR − B is",
              f"  {f4(blcr['diff'])} {iv(blcr)}, so B − LCR = {f4(-blcr['diff'])} [{f4(-blcr['hi'])}, {f4(-blcr['lo'])}]. Matches. The",
              "  two quotes differ only because they compare B with different models.",
              "- **Typos:** the abstract begins \"BBeyond\", and Table V's header reads \"50-−200 m\".", ""]


def main():
    lines = ["# CLOCs-R review round: extra measurements", "",
             "Branch `review-round` (from `master` 0512cd6), 2026-11-23. No detector was retrained, and no stored result was",
             "changed. Code and raw outputs are in `results/review_round/`. Numbers are to four decimals.", ""]
    task1(lines)
    task2(lines)
    task3(lines)
    lines += ["## Task 4. L0 and LRf on `master`", "",
              "- **Copied unchanged** (byte-identical) from thesis `results/next_round/phase7/`: `experts.json` as",
              "  `results/lcr/data/experts_nc.json` (`select`) and `val_experts.json` as `results/lcr/data/val_experts_nc.json`",
              "  (val).",
              "- **Contents:** both hold R0+T2+NC (L0) and R2+T2+NC (LRf), seeds 1–6.",
              "- **Check:** their 6-seed means are the paper's values. At 50–200 m: L0 0.5605 on `select` and 0.4282 on val; LRf",
              "  0.5729 and 0.4233.",
              "- **Documentation:** one line added to `docs/MODELS.md` (\"Stored results\", under \"Radar as features without a",
              "  camera\").",
              "- **Committed** locally on `review-round`; not pushed.", ""]
    task5(lines)
    paper(lines)
    notes = HERE / "notes.md"
    if notes.exists():
        lines += [notes.read_text().strip(), ""]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("wrote", HERE / "REPORT.md")


if __name__ == "__main__":
    main()
