#!/usr/bin/env bash
# Phase 7 of results/next_round/PLAN.md, up to the select results (val is scored once afterwards, by phase7_val.sh, when
# every model is frozen). One heavy job at a time, data-loader workers capped at 6. Steps whose output exists are
# skipped, so the chain can be restarted. From the repo root:
#   nohup setsid bash results/next_round/phase7_run.sh >> runs/next_round/phase7/chain.log 2>&1 &
cd "$(dirname "$0")/../.."
source env.sh
set -o pipefail
P1=runs/next_round/phase1
R=runs/next_round/phase7
O=results/next_round/phase7
C="--cells $P1/cells_in --cand3d $P1/cand_in/3d --cand2d $P1/cand_in/2d --far_weight 2 --gate 0.5,0.02 --train_split fit"
mkdir -p $R $O
stamp() { echo "== $(date '+%F %T') $*"; }
fail() { stamp "FAILED: $*"; exit 1; }
has() { [ -f "$2" ] && python -c "import json,sys; d=json.load(open(sys.argv[2])); sys.exit(0 if len(d.get(sys.argv[1],{}).get('runs',[]))>=int(sys.argv[3]) else 1)" "$1" "$2" "$3"; }
tune() {  # $1 config, $2 out json, $3 weights, rest: extra flags; seeds 1-3 then 4-6, select only
  for s in 1,2,3 4,5,6; do
    local n=$([ $s = 1,2,3 ] && echo 3 || echo 6)
    has "$1" "$2" $n && continue
    local ok=0 try
    for try in 1 2 3; do  # the shared GPU has twice failed one process's first backward pass at start-up (CUDA "unspecified launch failure")
      stamp "fusion $1 seeds $s -> $2 (attempt $try)"
      if nice -n 5 python -W ignore -m clocs_r.tune_fusion $C --weights "$3" --configs "$1" --splits select --seeds $s "${@:4}" --out "$2" \
        >> $R/fusion.log 2>&1; then ok=1; break; fi
      stamp "attempt $try failed; retrying"
    done
    [ $ok = 1 ] || fail "fusion $1 $s"
  done
}
stamp "PHASE 7 chain START ($(df -h ~ | tail -1 | awk '{print $4}') free); waiting for the contribution ablations"
while pgrep -f "bash runs/next_round/contrib_run.sh" > /dev/null; do sleep 60; done

# 1. camera-free experts that need no radar detector
tune R0+T2+NC $O/experts.json $R/fusion
tune R2+T2+NC $O/experts.json $R/fusion

# 2. D2: motion-corrected radar + clutter probability, trained, exported, scored alone on select
D2=clocs_r/configs/pointpillar_radar_d2_aimotive.yaml
if [ ! -f $R/d2_pp/checkpoint_epoch_40.pth ]; then
  stamp "D2 training"
  nice -n 5 python -W ignore -m clocs_r.train_second --config $D2 --out $R/d2_pp --train_split fit --workers 6 --keep_every 40 --resume \
    2>&1 | grep -v Warning >> $R/d2_train.log || fail "D2 training"
fi
if [ ! -f $R/d2_dets/DONE ]; then
  stamp "D2 export"
  nice -n 5 python -W ignore -m clocs_r.export_3d --config $D2 --ckpt $R/d2_pp/checkpoint_epoch_40.pth --out $R/d2_dets --splits fit,select,val \
    --nms --batch 16 2>&1 | grep -v Warning >> $R/d2_export.log || fail "D2 export"
  touch $R/d2_dets/DONE
fi
[ -f $O/radar_detector_d2.json ] || nice -n 5 python -W ignore -m clocs_r.radar_candidates --detector select --dets $R/d2_dets --tag d2 \
  --out $O/radar_detector_d2.json >> $R/detectors.log 2>&1 || fail "D2 alone"

# 3. D3: RadarDistill teacher (10 epochs) and student (20 epochs), by the 24-hour rule; radar input = better of D1, D2
INPUT=$(python results/next_round/phase7_decide.py radar_input) || fail "radar input"
stamp "RadarDistill radar input: $INPUT"
[ -f $R/rd_teacher/checkpoint_epoch_10.pth ] || { stamp "RadarDistill teacher";
  nice -n 5 python -W ignore -m clocs_r.radardistill_aimotive --stage teacher --out $R/rd_teacher --epochs 10 --radar_input $INPUT \
    2>&1 | grep -v -i warn >> $R/rd_teacher.log || fail "teacher"; }
[ -f $R/rd_student/checkpoint_epoch_20.pth ] || { stamp "RadarDistill student";
  nice -n 5 python -W ignore -m clocs_r.radardistill_aimotive --stage student --init $R/rd_teacher/checkpoint_epoch_10.pth --out $R/rd_student \
    --epochs 20 --radar_input $INPUT 2>&1 | grep -v -i warn >> $R/rd_student.log || fail "student"; }
if [ ! -f $R/d3_dets/DONE ]; then
  stamp "RadarDistill export"
  nice -n 5 python -W ignore -m clocs_r.radardistill_aimotive --stage export --ckpt $R/rd_student/checkpoint_epoch_20.pth --out $R/d3_dets \
    --splits fit,select,val --radar_input $INPUT 2>&1 | grep -v -i warn >> $R/d3_export.log || fail "D3 export"
  touch $R/d3_dets/DONE
fi
[ -f $O/radar_detector_d3.json ] || nice -n 5 python -W ignore -m clocs_r.radar_candidates --detector select --dets $R/d3_dets --tag d3 \
  --out $O/radar_detector_d3.json >> $R/detectors.log 2>&1 || fail "D3 alone"

# 4. LRd with the best detector on select
DETS=$(python results/next_round/phase7_decide.py detector) || fail "detector choice"
stamp "radar detector for RP: $DETS"
echo "$DETS" > $O/rp_detector.txt
tune R0+T2+RP $O/experts_rp.json $R/fusion_rp --radar_dets $DETS

# 5. the selector with the better radar expert (LRf or LRd) on select
read LR LRW < <(python results/next_round/phase7_decide.py radar_expert) || fail "radar expert"
stamp "radar expert for the selector: $LR ($LRW)"
echo "$LR $LRW" > $O/radar_expert.txt
python -c "import json,sys; d=json.load(open(sys.argv[1])) if __import__('os').path.exists(sys.argv[1]) else {}; sys.exit(0 if len(d.get('S('+sys.argv[2]+')',{}).get('runs',[]))==6 else 1)" \
  $O/selector.json "$LR" || { stamp "selector";
  nice -n 5 python -W ignore -m clocs_r.selector --radar_expert "$LR" --radar_weights "$LRW" --radar_dets "$DETS" --seeds 1,2,3,4,5,6 \
    --splits select --processes 3 --out $O/selector.json >> $R/selector.log 2>&1 || fail "selector"; }
CUR=$(python results/next_round/phase7_decide.py final) || fail "final rule"
stamp "current model by the rule: $CUR"
stamp "PHASE 7 select DONE ($(df -h ~ | tail -1 | awk '{print $4}') free); val is scored once by phase7_val.sh"
