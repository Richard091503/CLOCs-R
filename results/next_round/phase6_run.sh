#!/usr/bin/env bash
# Phase 6 of results/next_round/PLAN.md: the radar PointPillars (trained by runs/next_round/phase6/train_radar.sh, which
# this waits for), its boxes on fit/select/val, the detector on its own, the radar-sourced candidates, R2+T2+K40+G+RD by
# the rule, its 6-seed final on select and val, and the extras. Steps whose output exists are skipped. From the repo root:
#   nohup setsid bash results/next_round/phase6_run.sh >> runs/next_round/phase6/chain.log 2>&1 &
cd "$(dirname "$0")/../.."
source env.sh
set -o pipefail
P1=runs/next_round/phase1
R=runs/next_round/phase6
O=results/next_round/phase6
C="--cells $P1/cells_in --cand3d $P1/cand_in/3d --cand2d $P1/cand_in/2d --far_weight 2 --gate 0.5,0.02 --train_split fit"
CKPT=$R/radar_pp/checkpoint_epoch_40.pth
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

stamp "PHASE 6 chain START ($(df -h ~ | tail -1 | awk '{print $4}') free); waiting for the radar detector"
while pgrep -f train_radar.sh > /dev/null; do sleep 60; done
[ -f "$CKPT" ] || fail "no $CKPT"
stamp "radar detector trained"

if [ ! -f "$R/radar_dets/DONE" ]; then
  stamp "export radar boxes (fit, select, val)"
  python -W ignore -m clocs_r.export_3d --config clocs_r/configs/pointpillar_radar_aimotive.yaml --ckpt "$CKPT" --out "$R/radar_dets" \
    --splits fit,select,val --nms --batch 16 2>&1 | grep -v Warning >> "$R/export.log" || fail "export"
  touch "$R/radar_dets/DONE"
fi
stamp "radar-sourced candidates and the detector alone ($(df -h ~ | tail -1 | awk '{print $4}') free)"
python -W ignore -m clocs_r.radar_candidates --split fit,select,val >> "$R/candidates.log" 2>&1 || fail "radar candidates"
[ -f "$O/radar_detector.json" ] || python -W ignore -m clocs_r.radar_candidates --detector select,val --out "$O/radar_detector.json" \
  >> "$R/candidates.log" 2>&1 || fail "detector alone"

RD=R2+T2+K40+G+RD
tune "$RD" "$O/candidates.json" select 1,2,3 "$R/fusion"
tune "$RD" "$O/nocam.json" select 1,2,3 "$R/fusion" --from_weights --nocam
CURRENT=$(python results/next_round/phase6_decide.py) || fail "rule"
stamp "current model by the rule: $CURRENT"

tune "$RD" "$O/final_s123.json" select,val 1,2,3 "$R/fusion" --from_weights
tune "$RD" "$O/final_s456.json" select,val 4,5,6 "$R/fusion"
stamp "extras: SECOND ∪ radar, recall gained, false alarms"
python -W ignore results/next_round/phase6_extra.py --out "$O/extra.json" >> "$R/extra.log" 2>&1 || fail "extras"
stamp "PHASE 6 DONE ($(df -h ~ | tail -1 | awk '{print $4}') free)"
