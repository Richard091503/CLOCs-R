#!/usr/bin/env bash
# Phase 1 of results/next_round/PLAN.md (addendum): detectors on fit, cross-fitted detectors on the folds of fit,
# fusion on select with in-sample and cross-fitted training inputs, the rule, then val once for the chosen arm.
# Every step skips work whose output exists, so the script can be run again after an interruption. From the repo root:
#   nohup setsid bash results/next_round/phase1_run.sh >> runs/next_round/phase1/chain.log 2>&1 &
cd "$(dirname "$0")/../.."
source env.sh
set -o pipefail
R=runs/next_round/phase1
O=results/next_round/phase1
F="--configs R2+T2,R0+T2 --far_weight 2 --gate 0.5,0.02 --train_split fit"
mkdir -p "$R" "$O"
stamp() { echo "== $(date '+%F %T') $*"; }
fail() { stamp "FAILED: $*"; exit 1; }
disk_ok() { local free; free=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ "$free" -ge 10 ] || fail "only ${free} GB free before: $1"; }
# mmcv checkpoints carry the optimizer state too; inference needs only the weights
strip() { python -c "import torch,sys; c=torch.load(sys.argv[1],map_location='cpu'); c.pop('optimizer',None); torch.save(c,sys.argv[1])" "$1"; }

detectors() {  # $1 name, $2 training split
  local D="$R/det/$1"
  mkdir -p "$D/second" "$D/cascade"
  if ! grep -q "training finished" "$D/second/train.log" 2>/dev/null; then
    stamp "SECOND on $2"
    python -W ignore -m clocs_r.train_second --out "$D/second" --workers 6 --train_split "$2" --keep_every 10 --resume \
      > "$D/second/console.log" 2>&1 || fail "SECOND $1"
  fi
  if [ ! -f "$D/cascade/epoch_4.pth" ]; then
    stamp "Cascade R-CNN on $2"
    python -W ignore -m clocs_r.cascade_2d train --out "$D/cascade" --epochs 4 --accum 2 --train_split "$2" \
      > "$D/cascade/console.log" 2>&1 || fail "Cascade $1"
    strip "$D/cascade/epoch_4.pth" || fail "strip $1"
  fi
}

export_cands() {  # $1 detector name, $2 splits, $3 candidate dir
  local D="$R/det/$1"
  disk_ok "candidates of $1 on $2"
  [ -f "$3/3d/.done_$1" ] || { stamp "SECOND $1 candidates on $2"; python -W ignore -m clocs_r.export_3d --ckpt "$D/second/checkpoint_epoch_40.pth" \
      --out "$3/3d" --splits "$2" > "$R/export3d_$1.log" 2>&1 && touch "$3/3d/.done_$1"; } || fail "export 3d $1"
  [ -f "$3/2d/.done_$1" ] || { stamp "Cascade $1 candidates on $2"; python -W ignore -m clocs_r.cascade_2d export --out "$D/cascade" \
      --ckpt "$D/cascade/epoch_4.pth" --candidates "$3/2d" --splits "$2" > "$R/export2d_$1.log" 2>&1 && touch "$3/2d/.done_$1"; } || fail "export 2d $1"
}

IN=$R/cand_in      # fit detectors on fit, select and val
OOF=$R/cand_oof    # fold detectors on their held-out fold; select and val linked from $IN
CI=$R/cells_in
CO=$R/cells_oof

stamp "PHASE 1 START ($(df -h ~ | tail -1 | awk '{print $4}') free)"
detectors fit fit
export_cands fit select,val,fit "$IN"
[ -f "$O/detectors_fit/aimotive_second.json" ] || { stamp "detectors alone (fit detectors) on select and val"
  python -W ignore -m clocs_r.eval_detectors --splits select,val --out "$O/detectors_fit" --cand3d "$IN/3d" --cand2d "$IN/2d" \
    > "$R/eval_detectors_fit.log" 2>&1 || fail "eval detectors"; }

if [ ! -f "$O/select_insample.json" ]; then
  disk_ok "in-sample cells"; stamp "cells, in-sample"
  python -W ignore -m clocs_r.cache_cells --split select,val,fit --out "$CI" --cand3d "$IN/3d" --cand2d "$IN/2d" > "$R/cells_in.log" 2>&1 || fail "cells in"
  stamp "fusion, in-sample, seeds 1-3, on select"
  python -W ignore -m clocs_r.tune_fusion $F --splits select --seeds 1,2,3 --cells "$CI" --cand3d "$IN/3d" --cand2d "$IN/2d" \
    --weights "$R/fusion_insample" --out "$O/select_insample.json" > "$R/fusion_insample.log" 2>&1 || fail "fusion in-sample"
fi

for k in 0 1 2; do detectors "fold$k" "notfold$k"; done
rm -f "$CI"/fit_*  # in-sample fit cells: 4.8 GB, rebuilt in a few minutes if that arm is chosen
mkdir -p "$OOF/3d" "$OOF/2d" "$CO"
python - "$IN" "$OOF" <<'EOF' || fail "links"
import os, sys
from clocs_r import splits
src, dst = sys.argv[1:]
for s in splits.scenes("select") + splits.scenes("val"):
    name = os.path.basename(s.rstrip("/"))
    for d in ("3d", "2d"):
        link = os.path.join(dst, d, name)
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(os.path.join(src, d, name)), link)
EOF
for k in 0 1 2; do export_cands "fold$k" "fold$k" "$OOF"; done
for f in "$CI"/select_* "$CI"/val_*; do [ -e "$CO/$(basename "$f")" ] || ln -s "$(realpath "$f")" "$CO/"; done

if [ ! -f "$O/select_crossfit.json" ]; then
  disk_ok "cross-fitted cells"; stamp "cells, cross-fitted"
  python -W ignore -m clocs_r.cache_cells --split fit --out "$CO" --cand3d "$OOF/3d" --cand2d "$OOF/2d" > "$R/cells_oof.log" 2>&1 || fail "cells oof"
  stamp "fusion, cross-fitted, seeds 1-3, on select"
  python -W ignore -m clocs_r.tune_fusion $F --splits select --seeds 1,2,3 --cells "$CO" --cand3d "$OOF/3d" --cand2d "$OOF/2d" \
    --weights "$R/fusion_crossfit" --out "$O/select_crossfit.json" > "$R/fusion_crossfit.log" 2>&1 || fail "fusion cross-fitted"
fi

ARM=$(python results/next_round/phase1_decide.py) || fail "rule"
stamp "rule chose: $ARM"
if [ "$ARM" = crossfit ]; then C=$CO; CAND=$OOF; else C=$CI; CAND=$IN; rm -f "$CO"/fit_*; fi
if [ "$ARM" = insample ] && [ ! -f "$CI/fit_base.pkl" ]; then
  disk_ok "in-sample cells again"; stamp "cells, in-sample, again"
  python -W ignore -m clocs_r.cache_cells --split fit --out "$CI" --cand3d "$IN/3d" --cand2d "$IN/2d" > "$R/cells_in2.log" 2>&1 || fail "cells in again"
fi
W=$R/fusion_$ARM
stamp "val, once: $ARM, seeds 1-3 from their weights, seeds 4-6 trained"
[ -f "$O/final_s123.json" ] || python -W ignore -m clocs_r.tune_fusion $F --splits select,val --seeds 1,2,3 --from_weights --cells "$C" \
    --cand3d "$CAND/3d" --cand2d "$CAND/2d" --weights "$W" --out "$O/final_s123.json" > "$R/final_s123.log" 2>&1 || fail "final s1-3"
[ -f "$O/final_s456.json" ] || python -W ignore -m clocs_r.tune_fusion $F --splits select,val --seeds 4,5,6 --cells "$C" \
    --cand3d "$CAND/3d" --cand2d "$CAND/2d" --weights "$W" --out "$O/final_s456.json" > "$R/final_s456.log" 2>&1 || fail "final s4-6"
stamp "PHASE 1 DONE ($(df -h ~ | tail -1 | awk '{print $4}') free)"
