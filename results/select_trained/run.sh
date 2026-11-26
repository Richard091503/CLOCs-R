#!/usr/bin/env bash
# Fusion networks trained on select (results/select_trained/PREDICTIONS.md), end to end, one job at a time. Every step
# skips work whose output exists, so the script can be rerun after an interruption. From the repository root:
#   nohup setsid bash results/select_trained/run.sh >> runs/select_trained/run.log 2>&1 &
# Needs the cached cells, candidates and D2 boxes (runs/lcr), the paper's fit-trained networks (runs/next_round), and
# the thesis worktree beside this checkout (../CLOCs-R-thesis: B's fit-trained weights, and the equivalence check).
cd "$(dirname "$0")/../.."
source env.sh
set -o pipefail
R=runs/lcr
S=runs/select_trained
O=results/select_trained
F="--cells $R/cells --cand3d $R/cand/3d --cand2d $R/cand/2d --radar_dets $R/d2_dets"
FIT_STEPS=53979  # 3 epochs x 17,993 fit frames, as every fit-trained fusion network of the paper
stamp() { echo "== $(TZ=America/New_York date '+%F %T %Z') $*"; echo -e "$(date +%s)\t$*" >> $S/times.tsv; }
fail() { stamp "FAILED: $*"; exit 1; }
# Some CUDA start-ups failed on the shared GPU; a failed step is retried (as run_lcr.sh)
retry() { for a in 1 2 3; do "$@" && return 0; stamp "attempt $a failed: retrying in 60 s"; sleep 60; done; return 1; }
mkdir -p $S/step0 $S/primary $S/secondary $S/equivalence
stamp "SELECT-TRAINED START ($(df -h ~ | tail -1 | awk '{print $4}') free)"

# Step 0: retrain fit-trained CLOCs-R seed 1 with this code; its val APCD at 50-200 m must match the stored value
if [ ! -f $O/step0.json ]; then
  stamp "step 0 start"
  retry python -W ignore -m clocs_r.tune_fusion $F --weights $S/step0 --configs R0+T2+K40+RC --splits val --seeds 1 \
    --out $S/step0_val.json >> $S/step0.log 2>&1 || fail "step 0 training"
  python - <<'PY' || fail "step 0: no match"
import json, sys
here = json.load(open("runs/select_trained/step0_val.json"))["R0+T2+K40+RC"]["runs"][0]["val"]
ref = [r for r in json.load(open("results/lcr/data/lcr_val.json"))["R0+T2+K40+RC"]["runs"] if r["seed"] == 1][0]["val"]
diffs = {g: {b: here[g][b]["APCD"] - ref[g][b]["APCD"] for b in ref[g]} for g in ref}
out = dict(rule="retrained fit CLOCs-R (R0+T2+K40+RC) seed 1, val APCD at 50-200 m, equal to the stored value to 3 decimals",
           here=here["overall"]["50-200"]["APCD"], stored=ref["overall"]["50-200"]["APCD"],
           max_abs_diff_any_bin_or_condition=max(abs(v) for g in diffs.values() for v in g.values()))
out["match_3_decimals"] = round(out["here"], 3) == round(out["stored"], 3)
json.dump(out, open("results/select_trained/step0.json", "w"), indent=1)
print(json.dumps(out))
sys.exit(0 if out["match_3_decimals"] else 1)
PY
  stamp "step 0 done: $(cat $O/step0.json | tr -d '\n ')"
fi

# Master's code (NC token, --steps, blend.py) trains exactly as the thesis code that made the fit-trained networks
if [ ! -f $O/equivalence.json ]; then
  stamp "equivalence check start"
  ( source ../CLOCs-R-thesis/env.sh && python -W ignore $O/check_equivalence.py thesis $S/equivalence ) \
    >> $S/equivalence.log 2>&1 || fail "equivalence: thesis side"
  python -W ignore $O/check_equivalence.py master $S/equivalence >> $S/equivalence.log 2>&1 || fail "equivalence: master side"
  python -W ignore $O/check_equivalence.py compare $S/equivalence $O/equivalence.json >> $S/equivalence.log 2>&1 \
    || fail "equivalence: the two code bases differ ($O/equivalence.json)"
  stamp "equivalence check done: identical"
fi
# CHECKS_ONLY=1: stop here (step 0 and the equivalence check train nothing on select; PREDICTIONS.md must be
# committed before the training below starts)
[ -n "$CHECKS_ONLY" ] && { stamp "checks done; stopping before any training on select (CHECKS_ONLY)"; exit 0; }

train() {  # length, steps option, config
  local len=$1 st=$2 c=$3
  for s in 1,2,3 4,5,6; do
    [ -f $S/$len/${c}_s${s: -1}.pth ] && continue
    stamp "$len $c seeds $s start"
    retry python -W ignore -m clocs_r.tune_fusion $F --weights $S/$len --configs $c --train_split select $st --splits select \
      --seeds $s --out $S/train_$len.json >> $S/train_$len.log 2>&1 || fail "$len $c $s"
    stamp "$len $c seeds $s done"
  done
}
blend_runs() {  # length
  for s in 1,2,3 4,5,6; do
    [ -f $S/$1/blend_s${s: -1}.pth ] && continue
    stamp "$1 B_sel seeds $s start"
    retry python -W ignore $O/blend.py --length $1 --seeds $s --out $S/$1 --log $S/train_$1_blend.json >> $S/train_$1.log 2>&1 \
      || fail "$1 B_sel $s"
    stamp "$1 B_sel seeds $s done"
  done
}

# Primary: the fit runs' 53,979 optimizer steps on select. Secondary: 3 epochs over select (10,227 steps).
for c in R0+T2+NC R0+T2+RP R0+T2+K40 R0+T2+K40+RC; do train primary "--steps $FIT_STEPS" $c; done
blend_runs primary
for c in R0+T2+NC R0+T2+RP R0+T2+K40 R0+T2+K40+RC; do train secondary "" $c; done
blend_runs secondary
stamp "TRAINING DONE ($(df -h ~ | tail -1 | awk '{print $4}') free)"

# Val, once for every model, after all training
[ -f $O/val.json ] || { stamp "val start"; python -W ignore $O/score_val.py --processes 5 --out $O/val.json >> $S/val.log 2>&1 || fail "val"; }
stamp "SELECT-TRAINED DONE ($(df -h ~ | tail -1 | awk '{print $4}') free)"
