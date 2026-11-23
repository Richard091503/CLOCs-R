#!/usr/bin/env bash
# Phase 7 of results/next_round/PLAN.md, after "PHASE 7 select DONE" and the user's go-ahead: val scored ONCE for every
# Phase 7 model from its frozen weights, then scene-bootstrap intervals on select and val (REEVALUATION.md section 4).
# First the post-hoc fixed-blend ablation B (PLAN.md, Phase 7 blend addendum) is trained on fit and scored on select.
#   nohup setsid bash results/next_round/phase7_val.sh >> runs/next_round/phase7/val.log 2>&1 &
cd "$(dirname "$0")/../.."
source env.sh
set -o pipefail
P1=runs/next_round/phase1
R=runs/next_round/phase7
O=results/next_round/phase7
C="--cells $P1/cells_in --cand3d $P1/cand_in/3d --cand2d $P1/cand_in/2d --far_weight 2 --gate 0.5,0.02 --train_split fit"
DETS=$(cat $O/rp_detector.txt)
read LR LRW < $O/radar_expert.txt
stamp() { echo "== $(date '+%F %T') $*"; }
fail() { stamp "FAILED: $*"; exit 1; }
retry() { for a in 1 2 3; do "$@" && return 0; stamp "attempt $a failed: retrying in 60 s"; sleep 60; done; return 1; }
grep -q "PHASE 7 select DONE" $R/chain.log || fail "the select stage has not finished"
if ! python -c "import json,sys; sys.exit(len(json.load(open('$O/blend.json'))['B($LR)']['runs']) != 6)" 2>/dev/null; then
  stamp "BLEND select START"
  retry nice -n 5 python -W ignore -m clocs_r.selector --blend --radar_expert "$LR" --radar_weights "$LRW" --radar_dets "$DETS" \
    --seeds 1,2,3,4,5,6 --splits select --processes 3 --out $O/blend.json >> $R/blend.log 2>&1 || fail "blend select"
fi
stamp "BLEND select DONE: $(tail -1 $R/blend.log)"
stamp "PHASE 7 val START ($(df -h ~ | tail -1 | awk '{print $4}') free)"
for s in 1,2,3 4,5,6; do
  retry nice -n 5 python -W ignore -m clocs_r.tune_fusion $C --weights $R/fusion --configs R0+T2+NC,R2+T2+NC --splits val --seeds $s --from_weights \
    --out $O/val_experts.json >> $R/val_fusion.log 2>&1 || fail "val L0, LRf $s"
  retry nice -n 5 python -W ignore -m clocs_r.tune_fusion $C --weights $R/fusion_rp --configs R0+T2+RP --splits val --seeds $s --from_weights \
    --radar_dets $DETS --out $O/val_experts_rp.json >> $R/val_fusion.log 2>&1 || fail "val LRd $s"
done
retry nice -n 5 python -W ignore -m clocs_r.selector --radar_expert "$LR" --radar_weights "$LRW" --radar_dets "$DETS" --seeds 1,2,3,4,5,6 \
  --splits val --processes 3 --from_weights --out $O/selector_val.json >> $R/val_selector.log 2>&1 || fail "val selector"
retry nice -n 5 python -W ignore -m clocs_r.selector --blend --radar_expert "$LR" --radar_weights "$LRW" --radar_dets "$DETS" --seeds 1,2,3,4,5,6 \
  --splits val --processes 3 --from_weights --out $O/blend_val.json >> $R/val_selector.log 2>&1 || fail "val blend"
stamp "val scored; bootstrap"
# the best single expert on select (L, LC or LR*), for RQ3
BEST=$(python - "$LR" <<'PY'
import json, sys, numpy as np
lr = sys.argv[1]
sec = json.load(open("results/next_round/phase1/detectors_fit/aimotive_second.json"))["select"]["overall"]["50-200"]["APCD"]
def mean(name, files):
    rs = [r for f in files for r in json.load(open(f)).get(name, {}).get("runs", [])]
    return np.mean([r["select"]["overall"]["50-200"]["APCD"] for r in rs])
m = {"L": sec, "LC": mean("R0+T2+K40", ["results/next_round/phase3/final_r0_s123.json", "results/next_round/phase3/final_s456.json"]),
     "LR": mean(lr, ["results/next_round/phase7/experts.json", "results/next_round/phase7/experts_rp.json"])}
print(max(m, key=m.get))
PY
) || fail "best expert"
stamp "best single expert on select: $BEST"
MODELS='[["L","second","",""],["L0","fusion","R0+T2+NC","'$R'/fusion"],["LC","fusion","R0+T2+K40","runs/next_round/phase3/fusion"],
["LRf","fusion","R2+T2+NC","'$R'/fusion"],["LRd","fusion","R0+T2+RP","'$R'/fusion_rp"],["S","selector","'$LR'|'$LRW'","'$O'"],["B","blend","'$LR'|'$LRW'","'$O'"],
["inc","fusion","R2+T2+K40+G","runs/next_round/phase4/fusion"]]'
LRLAB=$([ "$LR" = R0+T2+RP ] && echo LRd || echo LRf)
BESTLAB=$([ "$BEST" = LR ] && echo $LRLAB || echo $BEST)
COMP='[["LRf","L0"],["LRd","L0"],["LRf","L"],["LRd","L"],["LRd","LRf"],["S","'$BESTLAB'"],["S","L"],["S","inc"],["LC","L"],["S","B"],["B","'$BESTLAB'"],["B","L"],["B","inc"]]'
for split in select val; do
  retry nice -n 5 python -W ignore -m clocs_r.bootstrap --split $split --models "$MODELS" --comparisons "$COMP" --radar_dets $DETS \
    --out $O/bootstrap_$split.json >> $R/bootstrap.log 2>&1 || fail "bootstrap $split"
done
stamp "PHASE 7 val DONE ($(df -h ~ | tail -1 | awk '{print $4}') free)"
