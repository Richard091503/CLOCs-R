#!/bin/bash
# Review round, task 5: LRd (R0+T2+RP) at pairing radius 1, 2 (reproduction check of the stored LRd) and 3 m, seeds 1-3,
# trained on fit and scored on select, then scored once on val from the saved networks.
#   nohup setsid bash ~/Documents/CLOCs-R-review/results/review_round/run_radius.sh >> ~/Documents/CLOCs-R/runs/review_round/radius.log 2>&1 &
cd ~/Documents/CLOCs-R || exit 1
source env.sh
OUT=../CLOCs-R-review/results/review_round/radius
mkdir -p $OUT
stamp() { echo "$(date -u '+%F %H:%M UTC') $*"; }
disk_ok() { local a=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ "$a" -ge 5 ] || { stamp "STOP: $a GB free"; exit 1; }; }
CI=runs/next_round/phase1/cells_in
if [ ! -f $CI/fit_base.pkl ]; then  # the base cells deleted in the 2026-11-23 cleanup; rebuilt byte-identical
  disk_ok; stamp "rebuild fit base cells"
  python -W ignore -m clocs_r.cache_cells --split fit --no_radar --out $CI --cand3d $CI/../cand_in/3d --cand2d $CI/../cand_in/2d || exit 1
fi
for r in 2.0 1.0 3.0; do
  C=runs/review_round/cells_r$r
  mkdir -p $C
  for f in $CI/*.pkl; do case $f in *_rp_*) ;; *) ln -sf "$(readlink -f $f)" $C/ ;; esac; done  # own folder for the new pair caches
  W=runs/review_round/fusion_r$r
  disk_ok
  if [ ! -f $OUT/lrd_r$r.json ]; then
    stamp "train LRd radius $r m, seeds 1-3, score select"
    python -W ignore ../CLOCs-R-review/results/review_round/radius_lrd.py --radius $r --configs R0+T2+RP --seeds 1,2,3 --splits select \
      --cells $C --weights $W --radar_dets runs/lcr/d2_dets --out $OUT/lrd_r$r.json || { stamp "FAILED $r select"; exit 1; }
  fi
  if [ ! -f $OUT/lrd_r${r}_val.json ]; then
    stamp "score LRd radius $r m on val (frozen)"
    python -W ignore ../CLOCs-R-review/results/review_round/radius_lrd.py --radius $r --configs R0+T2+RP --seeds 1,2,3 --splits val \
      --from_weights --cells $C --weights $W --radar_dets runs/lcr/d2_dets --out $OUT/lrd_r${r}_val.json || { stamp "FAILED $r val"; exit 1; }
  fi
done
stamp "RADIUS DONE"
