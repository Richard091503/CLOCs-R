#!/usr/bin/env bash
# Phase 3 of results/next_round/PLAN.md: visibility inputs and training changes on R2+T2, ablated on select by the
# rule written there (phase3_decide.py), then val once for the configuration the rule picked. Steps whose output
# exists are skipped. From the repo root:
#   nohup setsid bash results/next_round/phase3_run.sh >> runs/next_round/phase3/chain.log 2>&1 &
cd "$(dirname "$0")/../.."
source env.sh
set -o pipefail
P1=runs/next_round/phase1
R=runs/next_round/phase3
O=results/next_round/phase3
D=results/next_round/phase3_decide.py
C="--cells $P1/cells_in --cand3d $P1/cand_in/3d --cand2d $P1/cand_in/2d --far_weight 2 --gate 0.5,0.02 --train_split fit --weights $R/fusion"
mkdir -p "$R" "$O"
stamp() { echo "== $(date '+%F %T') $*"; }
fail() { stamp "FAILED: $*"; exit 1; }
has() { [ -f "$2" ] && python -c "import json,sys; sys.exit(0 if sys.argv[1] in json.load(open(sys.argv[2])) else 1)" "$1" "$2"; }
tune() {  # $1 configs, $2 out json, $3 splits, $4 seeds, rest: extra flags; skips configs already in the file
  local todo=() c
  for c in ${1//,/ }; do has "$c" "$2" || todo+=("$c"); done
  [ ${#todo[@]} -eq 0 ] && return 0
  local list; list=$(IFS=,; echo "${todo[*]}")
  stamp "fusion $list, seeds $4, on $3 -> $2"
  python -W ignore -m clocs_r.tune_fusion $C --configs "$list" --splits "$3" --seeds "$4" "${@:5}" --out "$2" \
    >> "$R/fusion.log" 2>&1 || fail "fusion $list"
}

stamp "PHASE 3 START ($(df -h ~ | tail -1 | awk '{print $4}') free)"
python -W ignore -m clocs_r.visibility --split select,val,fit --cells $P1/cells_in --cand3d $P1/cand_in/3d --cand2d $P1/cand_in/2d \
  >> "$R/visibility.log" 2>&1 || fail "visibility"

# Stage A: the reference (R2+T2 retrained, which must reproduce Phase 1) and each token alone
A="R2+T2"
for t in H P I1 I2 K40 D2 D3 S50 S100; do A="$A,R2+T2+$t"; done
tune "$A" "$O/stageA.json" select 1,2,3
KEPT=$(python $D stageA) || fail "stage A rule"
stamp "stage A kept: ${KEPT:-nothing}"

# Stage B: add the kept tokens one at a time, largest gain first
CUR=R2+T2
for t in $KEPT; do
  NEW="$CUR+$t"
  [ "$CUR" = R2+T2 ] || tune "$NEW" "$O/stageB.json" select 1,2,3
  if python $D beats "$NEW" "$CUR" 2>> "$R/decisions.log"; then CUR=$NEW; fi
done
stamp "final configuration: $CUR"
echo "$CUR" > "$O/final_config.txt"

if [ "$CUR" = R2+T2 ]; then
  stamp "nothing adopted: the final is R2+T2, already scored on val in Phase 1"
else
  R0=${CUR/R2+T2/R0+T2}
  tune "$CUR" "$O/final_s123.json" select,val 1,2,3 --from_weights
  tune "$R0" "$O/final_r0_s123.json" select,val 1,2,3
  tune "$CUR,$R0" "$O/final_s456.json" select,val 4,5,6
fi
stamp "PHASE 3 DONE ($(df -h ~ | tail -1 | awk '{print $4}') free)"
