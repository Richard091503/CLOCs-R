#!/usr/bin/env bash
# Phase 7 LCR addendum (results/next_round/PLAN.md): one CLOCs network over SECOND, Cascade R-CNN and D2 (R0+T2+K40+RC).
# Trains 6 seeds on fit and scores select, applies the rule, scores val once, and runs the scene bootstrap. Post-val
# development, labelled as such.
#   nohup setsid bash results/next_round/phase7_lcr.sh >> runs/next_round/phase7/lcr.log 2>&1 &
cd "$(dirname "$0")/../.."
source env.sh
set -o pipefail
P1=runs/next_round/phase1
R=runs/next_round/phase7
O=results/next_round/phase7
C="--cells $P1/cells_in --cand3d $P1/cand_in/3d --cand2d $P1/cand_in/2d --far_weight 2 --gate 0.5,0.02 --train_split fit"
DETS=$(cat $O/rp_detector.txt)
NAME=R0+T2+K40+RC
stamp() { echo "== $(date '+%F %T') $*"; }
fail() { stamp "FAILED: $*"; exit 1; }
retry() { for a in 1 2 3; do "$@" && return 0; stamp "attempt $a failed: retrying in 60 s"; sleep 60; done; return 1; }
stamp "LCR START ($(df -h ~ | tail -1 | awk '{print $4}') free)"
for s in 1,2,3 4,5,6; do
  retry nice -n 5 python -W ignore -m clocs_r.tune_fusion $C --weights $R/fusion_lcr --configs $NAME --splits select --seeds $s \
    --radar_dets $DETS --out $O/lcr.json >> $R/lcr_fusion.log 2>&1 || fail "train $NAME $s"
  stamp "trained and scored on select: seeds $s"
done
python - <<'PY' || fail "rule"
import json, numpy as np
O = "results/next_round/phase7"
lcr = json.load(open(f"{O}/lcr.json"))["R0+T2+K40+RC"]["runs"]
b = json.load(open(f"{O}/blend.json"))["B(R0+T2+RP)"]["runs"]
assert len(lcr) == 6 and len(b) == 6
out = {}
for bn in ("50-200", "100-200"):
    x = np.array([r["select"]["overall"][bn]["APCD"] for r in lcr])
    y = np.array([r["select"]["overall"][bn]["APCD"] for r in b])
    out[bn] = dict(lcr=float(x.mean()), lcr_sd=float(x.std(ddof=1)), b=float(y.mean()), passes=bool(x.mean() >= y.mean() - x.std(ddof=1)))
out["final"] = "R0+T2+K40+RC (LCR)" if all(out[bn]["passes"] for bn in ("50-200", "100-200")) else "B(R0+T2+RP)"
out["rule"] = "LCR is the final model if its select mean is not below B's by more than LCR's seed SD at 50-200 and 100-200 m"
json.dump(out, open(f"{O}/decision_lcr.json", "w"), indent=1)
print(json.dumps(out))
PY
stamp "rule on select: $(python -c "import json; print(json.load(open('$O/decision_lcr.json'))['final'])")"
for s in 1,2,3 4,5,6; do
  retry nice -n 5 python -W ignore -m clocs_r.tune_fusion $C --weights $R/fusion_lcr --configs $NAME --splits val --seeds $s --from_weights \
    --radar_dets $DETS --out $O/lcr_val.json >> $R/lcr_fusion.log 2>&1 || fail "val $NAME $s"
done
stamp "val scored: $(grep "R0+T2+K40+RC {'val'" $R/lcr_fusion.log | tail -1)"
MODELS='[["L","second","",""],["LC","fusion","R0+T2+K40","runs/next_round/phase3/fusion"],["LRd","fusion","R0+T2+RP","'$R'/fusion_rp"],
["B","blend","R0+T2+RP|'$R'/fusion_rp","'$O'"],["LCR","fusion","'$NAME'","'$R'/fusion_lcr"]]'
COMP='[["LCR","LC"],["LCR","LRd"],["LCR","B"],["LCR","L"]]'
for split in select val; do
  retry nice -n 5 python -W ignore -m clocs_r.bootstrap --split $split --models "$MODELS" --comparisons "$COMP" --radar_dets $DETS \
    --out $O/bootstrap_lcr_$split.json >> $R/lcr_bootstrap.log 2>&1 || fail "bootstrap $split"
  stamp "bootstrap $split done"
done
stamp "LCR DONE ($(df -h ~ | tail -1 | awk '{print $4}') free)"
