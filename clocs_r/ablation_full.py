"""The full-dataset ablation of results/full_dataset/ablation/PLAN.md: gate, feature sets, fusion changes, then val once.

    python -m clocs_r.ablation_full            # resumable: a stage whose output exists is not run again

Every training run goes through clocs_r.tune_fusion (far weight 2, 3 epochs). Selection reads holdout only; val is scored
in the last stage, after the final configurations are fixed. Decisions are written to ablation/decisions.json.
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np

from clocs_r import tune_fusion

OUT = pathlib.Path("results/full_dataset/ablation")
W = pathlib.Path("runs/ablation")
FIRST_ROUND = (pathlib.Path("results/full_dataset/selection.json"), pathlib.Path("runs/fusion_full"))
GATES = [(d, k) for d in (0.25, 0.5, 1.0) for k in (0.01, 0.02, 0.04)]
SETS = ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7")
CHANGES = ("T1", "T2", "T3", "T4a", "T4b", "T5")
RADAR_ONLY = {"T4a", "T4b", "T5"}
FAR_WEIGHT = 2.0
METRIC = ("overall", "50-200", "APCD")


def gate_tag(gate):
    return f"d{gate[0]:g}_k{gate[1]:g}"


def holdout_scores(entry):
    g, b, m = METRIC
    return [r["holdout"][g][b][m] for r in entry["runs"]]


def load(path):
    path = pathlib.Path(path)
    return json.load(open(path)) if path.exists() else {}


def reuse(names, gate, out_json, weights):
    """Copy first-round runs with identical settings (seeds 1-3, far weight 2, same gate) instead of retraining them."""
    src, src_w = FIRST_ROUND
    first, results = load(src), load(out_json)
    for name in names:
        e = first.get(name)
        if name in results or e is None or tuple(gate) != tune_fusion.GATE:  # the first round used the default gate only
            continue
        if e["settings"] != tune_fusion.parse(name) or e.get("far_weight") != FAR_WEIGHT:
            continue
        pathlib.Path(weights).mkdir(parents=True, exist_ok=True)
        for r in e["runs"]:
            shutil.copy(src_w / f"{name}_s{r['seed']}.pth", pathlib.Path(weights) / f"{name}_s{r['seed']}.pth")
        results[name] = dict(e, gate=list(gate), nocam=False, reused_from=str(src))
        print(f"reused {name} from {src}", flush=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=1)


def tune(configs, out_json, weights, gate, splits="holdout", seeds="1,2,3", from_weights=False, nocam=False):
    """Run tune_fusion for the configurations not yet in out_json (or all of them when scoring saved networks)."""
    have = load(out_json)
    todo = [c for c in configs if from_weights or c not in have]
    if not todo:
        return
    cmd = [sys.executable, "-W", "ignore", "-m", "clocs_r.tune_fusion", "--configs", ",".join(todo), "--out", str(out_json),
           "--weights", str(weights), "--far_weight", str(FAR_WEIGHT), "--gate", f"{gate[0]},{gate[1]}", "--splits", splits,
           "--seeds", seeds] + (["--from_weights"] if from_weights else []) + (["--nocam"] if nocam else [])
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def best(results, names):
    means = {n: float(np.mean(holdout_scores(results[n]))) for n in names}
    return max(means, key=means.get), means


def join(base, changes):
    return "+".join([base] + [c for c in CHANGES if c in changes])


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    decisions = load(OUT / "decisions.json")

    # 1. Gate, with R3
    gate_scores = {}
    for gate in GATES:
        out, w = OUT / f"gate_{gate_tag(gate)}.json", W / f"gate_{gate_tag(gate)}"
        reuse(["R3"], gate, out, w)
        tune(["R3"], out, w, gate)
        gate_scores[gate_tag(gate)] = holdout_scores(load(out)["R3"])
    gate = max(GATES, key=lambda g: np.mean(gate_scores[gate_tag(g)]))
    decisions["gate"] = dict(selected=list(gate), holdout={k: dict(mean=float(np.mean(v)), seeds=v) for k, v in gate_scores.items()})
    print("gate:", gate, flush=True)

    # 2. Feature sets at that gate
    out2, w2 = OUT / "feature_sets.json", W / f"sets_{gate_tag(gate)}"
    reuse(SETS, gate, out2, w2)
    if "R3" not in load(out2):  # R3 at this gate was trained in step 1
        r = load(out2)
        r["R3"] = load(OUT / f"gate_{gate_tag(gate)}.json")["R3"]
        w2.mkdir(parents=True, exist_ok=True)
        for s in (1, 2, 3):
            shutil.copy(W / f"gate_{gate_tag(gate)}" / f"R3_s{s}.pth", w2 / f"R3_s{s}.pth")
        json.dump(r, open(out2, "w"), indent=1)
    tune(SETS, out2, w2, gate)
    sets = load(out2)
    S, set_means = best(sets, SETS[1:])
    decisions["feature_set"] = dict(selected=S, holdout={n: dict(mean=float(np.mean(holdout_scores(sets[n]))), seeds=holdout_scores(sets[n])) for n in SETS})
    print("feature set:", S, flush=True)

    # 3. Fusion changes on top of S (same weights directory, so S itself is available)
    out3 = OUT / "changes.json"
    singles = [join(S, [c]) for c in CHANGES]
    reuse(singles, gate, out3, w2)
    tune(singles, out3, w2, gate)
    changes = load(out3)
    base = holdout_scores(sets[S])
    threshold = float(np.mean(base) + np.std(base, ddof=1))
    adopted = [c for c in CHANGES if np.mean(holdout_scores(changes[join(S, [c])])) > threshold]
    if "T4a" in adopted and "T4b" in adopted:
        worse = min(("T4a", "T4b"), key=lambda c: np.mean(holdout_scores(changes[join(S, [c])])))
        adopted.remove(worse)
    candidates = {S: sets[S], **{join(S, [c]): changes[join(S, [c])] for c in adopted}}
    combo = None
    if len(adopted) >= 2:
        combo = join(S, adopted)
        reuse([combo], gate, out3, w2)
        tune([combo], out3, w2, gate)
        candidates[combo] = load(out3)[combo]
        changes = load(out3)
    final, final_means = best(candidates, list(candidates))
    final_changes = [c for c in CHANGES if c in final.split("+")]
    twin = join("R0", [c for c in final_changes if c not in RADAR_ONLY])
    decisions["changes"] = dict(base=S, threshold=threshold, adopted=adopted, combination=combo,
                                holdout={n: dict(mean=float(np.mean(holdout_scores(e))), seeds=holdout_scores(e)) for n, e in changes.items()},
                                final=final, final_holdout=final_means)
    decisions["final"] = dict(clocs_r=final, clocs=twin, gate=list(gate))
    json.dump(decisions, open(OUT / "decisions.json", "w"), indent=1)
    print("final CLOCs-R:", final, "| final CLOCs:", twin, flush=True)

    # The twin, if it is not already a trained configuration
    out_twin = OUT / "twin.json"
    if twin not in sets and twin not in changes:
        reuse([twin], gate, out_twin, w2)
        tune([twin], out_twin, w2, gate)

    # 4. Val, once: every network of steps 2-3 and the twin, then seeds 4-6 for the two finals, camera removal, latency
    trained = list(SETS) + list(changes) + ([twin] if twin not in sets and twin not in changes else [])
    tune([n for n in trained if n not in load(OUT / "val.json")], OUT / "val.json", w2, gate, splits="holdout,val", from_weights=True)
    tune([final, twin], OUT / "extra_seeds.json", w2, gate, splits="holdout,val", seeds="4,5,6")
    tune([n for n in (final, twin) if n not in load(OUT / "val_nocam.json")], OUT / "val_nocam.json", w2, gate, splits="holdout,val",
         seeds="1,2,3,4,5,6", from_weights=True, nocam=True)
    if not (OUT / "latency.json").exists():
        subprocess.run([sys.executable, "-W", "ignore", "-m", "clocs_r.radar_latency", "--delta0", str(gate[0]), "--kappa", str(gate[1]),
                        "--fusion_dir", str(w2), "--out", str(OUT / "latency.json")], check=True)
    print("ABLATION DONE", flush=True)


if __name__ == "__main__":
    main()
