#!/usr/bin/env bash
# The full-dataset study of results/full_dataset/PLAN.md, after the detectors are trained:
#   candidates -> detector scores -> fusion cells -> fusion selection on holdout -> val, once.
# Waits for runs/second and runs/cascade to finish first. Run from the repository root:
#   nohup setsid bash clocs_r/run_full.sh > runs/run_full.log 2>&1 &
# Each step skips work whose output already exists, except the candidate export and the fusion runs.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
set -euo pipefail
cd "$CLOCS_ROOT"
OUT=results/full_dataset
W=runs/fusion_full
mkdir -p "$OUT" "$W" runs/candidates3d runs/candidates2d
stamp() { echo "== $(date '+%F %T') $*"; }

stamp "waiting for detector training"
until grep -q "training finished" runs/second/train.log 2>/dev/null && [ -f runs/cascade/epoch_4.pth ] && \
      ! pgrep -f "clocs_r[.]cascade_2d train" > /dev/null; do
  sleep 300
done

stamp "SECOND candidates (top 2,000 before NMS)"
[ -f runs/candidates3d/.done ] || { python -W ignore -m clocs_r.export_3d --ckpt runs/second/checkpoint_epoch_40.pth \
    --out runs/candidates3d --splits val,holdout,train && touch runs/candidates3d/.done; }
stamp "Cascade R-CNN candidates (top 200)"
[ -f runs/candidates2d/.done ] || { python -W ignore -m clocs_r.cascade_2d export --out runs/cascade --ckpt runs/cascade/epoch_4.pth \
    --candidates runs/candidates2d --splits val,holdout,train && touch runs/candidates2d/.done; }

stamp "detectors alone"
python -W ignore -m clocs_r.eval_detectors --splits holdout,val --out "$OUT/detectors"

stamp "fusion cells and radar features (gate 0.5 m, 0.02)"
python -W ignore -m clocs_r.cache_cells --split holdout,val,train

stamp "fusion selection on holdout (far weight 2)"
[ -f "$OUT/selection.json" ] || python -W ignore -m clocs_r.tune_fusion --configs R0,R0+T2,R2,R3,R2+T2,R3+T2 --far_weight 2 \
    --out "$OUT/selection.json" --weights "$W"
[ -f "$OUT/selection_far_weight_1.json" ] || python -W ignore -m clocs_r.tune_fusion --configs R0 --far_weight 1 \
    --out "$OUT/selection_far_weight_1.json" --weights "$W/far_weight_1"
python -W ignore -m clocs_r.select_full --selection "$OUT/selection.json" --out "$OUT/selected.json"

stamp "val, once, with the saved networks"
python -W ignore -m clocs_r.tune_fusion --from_weights --configs R0,R0+T2,R2,R3,R2+T2,R3+T2 --far_weight 2 --splits holdout,val \
    --out "$OUT/final.json" --weights "$W"
python -W ignore -m clocs_r.tune_fusion --from_weights --configs R0 --far_weight 1 --splits holdout,val \
    --out "$OUT/final_far_weight_1.json" --weights "$W/far_weight_1"
stamp "ALL DONE"
