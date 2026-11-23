#!/usr/bin/env bash
# Phase 4 of results/next_round/PLAN.md: the gated model with and without the condition flag, the rule with the
# camera-removal veto (phase4_decide.py), val once for the configuration it picked, and the gate plots. Steps whose
# output exists are skipped. From the repo root:
#   nohup setsid bash results/next_round/phase4_run.sh >> runs/next_round/phase4/chain.log 2>&1 &
cd "$(dirname "$0")/../.."
source env.sh
set -o pipefail
P1=runs/next_round/phase1
R=runs/next_round/phase4
O=results/next_round/phase4
C="--cells $P1/cells_in --cand3d $P1/cand_in/3d --cand2d $P1/cand_in/2d --far_weight 2 --gate 0.5,0.02 --train_split fit"
mkdir -p "$R" "$O"
stamp() { echo "== $(date '+%F %T') $*"; }
fail() { stamp "FAILED: $*"; exit 1; }
has() { [ -f "$2" ] && python -c "import json,sys; sys.exit(0 if sys.argv[1] in json.load(open(sys.argv[2])) else 1)" "$1" "$2"; }
tune() {  # $1 configs, $2 out json, $3 splits, $4 seeds, $5 weights, rest: extra flags; skips configs already in the file
  local todo=() c
  for c in ${1//,/ }; do has "$c" "$2" || todo+=("$c"); done
  [ ${#todo[@]} -eq 0 ] && return 0
  local list; list=$(IFS=,; echo "${todo[*]}")
  stamp "fusion $list, seeds $4, on $3 -> $2 ${*:6}"
  python -W ignore -m clocs_r.tune_fusion $C --weights "$5" --configs "$list" --splits "$3" --seeds "$4" "${@:6}" --out "$2" \
    >> "$R/fusion.log" 2>&1 || fail "fusion $list"
}

stamp "PHASE 4 START ($(df -h ~ | tail -1 | awk '{print $4}') free)"
python -W ignore -m clocs_r.visibility --split select,val,fit --cells $P1/cells_in --cand3d $P1/cand_in/3d --cand2d $P1/cand_in/2d \
  --moving 0.5,0.02 >> "$R/visibility.log" 2>&1 || fail "moving returns"

G=R2+T2+K40+G,R2+T2+K40+G+C
tune "$G" "$O/candidates.json" select 1,2,3 "$R/fusion"
tune "$G" "$O/nocam.json" select 1,2,3 "$R/fusion" --from_weights --nocam
tune R2+T2+K40 "$O/nocam_ref.json" select 1,2,3 runs/next_round/phase3/fusion --from_weights --nocam
FINAL=$(python results/next_round/phase4_decide.py) || fail "rule"
stamp "final configuration: $FINAL"

[ -f results/next_round/figures/gates_select.png ] || python -W ignore results/next_round/phase4_gates.py --configs "$G" --split select \
  --seeds 1,2,3 --out gates_select.png >> "$R/gates.log" 2>&1 || fail "gate plot, select"

if [ "$FINAL" = R2+T2+K40 ]; then
  stamp "nothing adopted: the final remains R2+T2+K40, already scored on val in Phase 3"
  cp results/next_round/figures/gates_select.png results/next_round/figures/gates.png
else
  R0=${FINAL/R2+T2/R0+T2}
  tune "$FINAL" "$O/final_s123.json" select,val 1,2,3 "$R/fusion" --from_weights
  tune "$R0" "$O/final_r0_s123.json" select,val 1,2,3 "$R/fusion"
  tune "$FINAL,$R0" "$O/final_s456.json" select,val 4,5,6 "$R/fusion"
  [ -f results/next_round/figures/gates.png ] || python -W ignore results/next_round/phase4_gates.py --configs "$FINAL" --split val \
    --seeds 1,2,3,4,5,6 --out gates.png >> "$R/gates.log" 2>&1 || fail "gate plot, val"
fi
stamp "PHASE 4 DONE ($(df -h ~ | tail -1 | awk '{print $4}') free)"
