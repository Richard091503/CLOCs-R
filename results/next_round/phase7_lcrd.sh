#!/usr/bin/env bash
# Phase 7 LCR-D addendum (results/next_round/PLAN.md): LCR with camera cells hidden beyond 50 m in training (token DF).
# Waits for the LCR run to finish (one heavy job at a time), trains 6 seeds on fit and scores select, applies the rule,
# scores val once, and runs the scene bootstrap. Post-val, val-informed development, labelled as such.
#   nohup setsid bash results/next_round/phase7_lcrd.sh >> runs/next_round/phase7/lcrd.log 2>&1 &
cd "$(dirname "$0")/../.."
source env.sh
set -o pipefail
P1=runs/next_round/phase1
R=runs/next_round/phase7
O=results/next_round/phase7
C="--cells $P1/cells_in --cand3d $P1/cand_in/3d --cand2d $P1/cand_in/2d --far_weight 2 --gate 0.5,0.02 --train_split fit"
DETS=$(cat $O/rp_detector.txt)
NAME=R0+T2+K40+RC+DF
stamp() { echo "== $(date '+%F %T') $*"; }
fail() { stamp "FAILED: $*"; exit 1; }
retry() { for a in 1 2 3; do "$@" && return 0; stamp "attempt $a failed: retrying in 60 s"; sleep 60; done; return 1; }
stamp "LCR-D queued: waiting for the LCR run"
until grep -q "LCR DONE\|FAILED" $R/lcr.log; do sleep 60; done
stamp "LCR-D START ($(df -h ~ | tail -1 | awk '{print $4}') free)"
for s in 1,2,3 4,5,6; do
  retry nice -n 5 python -W ignore -m clocs_r.tune_fusion $C --weights $R/fusion_lcrd --configs $NAME --splits select --seeds $s \
    --radar_dets $DETS --out $O/lcrd.json >> $R/lcrd_fusion.log 2>&1 || fail "train $NAME $s"
  stamp "trained and scored on select: seeds $s"
done
python - <<'PY' || fail "rule"
import json, numpy as np
O = "results/next_round/phase7"
d = json.load(open(f"{O}/lcrd.json"))["R0+T2+K40+RC+DF"]["runs"]
c = json.load(open(f"{O}/lcr.json"))["R0+T2+K40+RC"]["runs"]
assert len(d) == 6 and len(c) == 6
out = {}
for bn in ("0-30", "30-50", "50-200", "100-200"):
    x = np.array([r["select"]["overall"][bn]["APCD"] for r in d])
    y = np.array([r["select"]["overall"][bn]["APCD"] for r in c])
    out[bn] = dict(lcrd=float(x.mean()), lcrd_sd=float(x.std(ddof=1)), lcr=float(y.mean()), passes=bool(x.mean() >= y.mean() - x.std(ddof=1)))
out["final"] = "R0+T2+K40+RC+DF (LCR-D)" if all(out[bn]["passes"] for bn in ("0-30", "30-50", "50-200", "100-200")) else "R0+T2+K40+RC (LCR)"
out["rule"] = "LCR-D replaces LCR if its select mean is not below LCR's by more than LCR-D's seed SD in 0-30, 30-50, 50-200 and 100-200 m"
json.dump(out, open(f"{O}/decision_lcrd.json", "w"), indent=1)
print(json.dumps(out))
PY
stamp "rule on select: $(python -c "import json; print(json.load(open('$O/decision_lcrd.json'))['final'])")"
for s in 1,2,3 4,5,6; do
  retry nice -n 5 python -W ignore -m clocs_r.tune_fusion $C --weights $R/fusion_lcrd --configs $NAME --splits val --seeds $s --from_weights \
    --radar_dets $DETS --out $O/lcrd_val.json >> $R/lcrd_fusion.log 2>&1 || fail "val $NAME $s"
done
stamp "val scored: $(grep "R0+T2+K40+RC+DF {'val'" $R/lcrd_fusion.log | tail -1)"
MODELS='[["L","second","",""],["LC","fusion","R0+T2+K40","runs/next_round/phase3/fusion"],["LRd","fusion","R0+T2+RP","'$R'/fusion_rp"],
["LCR","fusion","R0+T2+K40+RC","'$R'/fusion_lcr"],["LCRD","fusion","'$NAME'","'$R'/fusion_lcrd"]]'
COMP='[["LCRD","LCR"],["LCRD","LC"],["LCRD","LRd"],["LCRD","L"]]'
for split in select val; do
  retry nice -n 5 python -W ignore -m clocs_r.bootstrap --split $split --models "$MODELS" --comparisons "$COMP" --radar_dets $DETS \
    --out $O/bootstrap_lcrd_$split.json >> $R/lcrd_bootstrap.log 2>&1 || fail "bootstrap $split"
  stamp "bootstrap $split done"
done
stamp "LCR-D DONE ($(df -h ~ | tail -1 | awk '{print $4}') free)"
