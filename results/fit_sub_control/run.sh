#!/usr/bin/env bash
# Size control for the select-trained result (results/fit_sub_control/PREDICTIONS.md), end to end, one job at a time.
# Every step skips work whose output exists, so the script can be rerun after an interruption. From the repository root:
#   nohup setsid bash results/fit_sub_control/run.sh >> runs/fit_sub_control/run.log 2>&1 &
# Same settings as the select-trained primary runs (results/select_trained/run.sh); the training frames are the fit
# scenes of subset.json (draw_subset.py) instead of select.
cd "$(dirname "$0")/../.."
source env.sh
set -o pipefail
R=runs/lcr
S=runs/fit_sub_control
O=results/fit_sub_control
F="--cells $R/cells --cand3d $R/cand/3d --cand2d $R/cand/2d --radar_dets $R/d2_dets"
FIT_STEPS=53979  # the select-trained primary runs' --steps (3 epochs x 17,993 fit frames)
stamp() { echo "== $(TZ=America/New_York date '+%F %T %Z') $*"; echo -e "$(date +%s)\t$*" >> $S/times.tsv; }
fail() { stamp "FAILED: $*"; exit 1; }
retry() { for a in 1 2 3; do "$@" && return 0; stamp "attempt $a failed: retrying in 60 s"; sleep 60; done; return 1; }
mkdir -p $S/primary
stamp "FIT-SUB START ($(df -h ~ | tail -1 | awk '{print $4}') free)"

# The predictions and the subset are fixed (committed, unchanged) before the first training run
git ls-files --error-unmatch $O/PREDICTIONS.md $O/subset.json > /dev/null 2>&1 || fail "PREDICTIONS.md and subset.json not committed"
git diff --quiet HEAD -- $O/PREDICTIONS.md $O/subset.json || fail "PREDICTIONS.md or subset.json changed since the commit"

for c in R0+T2+NC R0+T2+RP R0+T2+K40 R0+T2+K40+RC; do
  for s in 1,2,3 4,5,6; do
    [ -f $S/primary/${c}_s${s: -1}.pth ] && continue
    stamp "primary $c seeds $s start"
    retry python -W ignore -m clocs_r.tune_fusion $F --weights $S/primary --configs $c --train_split fit \
      --train_scenes $O/subset.json --steps $FIT_STEPS --splits select --seeds $s --out $S/train_primary.json \
      >> $S/train_primary.log 2>&1 || fail "primary $c $s"
    stamp "primary $c seeds $s done"
  done
done
# every network trained on exactly the subset's frames
python - <<'PY' || fail "training frames differ from subset.json"
import json, sys
want = json.load(open("results/fit_sub_control/subset.json"))["fit_sub"]["all"]["frames"]
got = {k: v.get("train_frames") for k, v in json.load(open("runs/fit_sub_control/train_primary.json")).items()}
print("training frames", got, "subset", want)
sys.exit(0 if all(g == want for g in got.values()) and len(got) == 4 else 1)
PY
stamp "TRAINING DONE ($(df -h ~ | tail -1 | awk '{print $4}') free)"

# Val, once for every model, after all training
[ -f $O/val.json ] || { stamp "val start"; python -W ignore $O/score_val.py --processes 5 --out $O/val.json >> $S/val.log 2>&1 || fail "val"; }
stamp "FIT-SUB DONE ($(df -h ~ | tail -1 | awk '{print $4}') free)"
