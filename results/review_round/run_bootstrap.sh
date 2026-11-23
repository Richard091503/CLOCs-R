#!/bin/bash
# Review round, task 1: scene- and recording-level bootstrap on select and val (cluster_bootstrap.py).
#   nohup setsid bash ~/Documents/CLOCs-R-review/results/review_round/run_bootstrap.sh >> ~/Documents/CLOCs-R/runs/review_round/bootstrap.log 2>&1 &
cd ~/Documents/CLOCs-R || exit 1
source ../CLOCs-R-thesis/env.sh
OUT=../CLOCs-R-review/results/review_round
for split in select val; do
  [ -f $OUT/bootstrap_$split.json ] && continue
  avail=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ "$avail" -ge 5 ] || { echo "STOP: ${avail} GB free"; exit 1; }
  echo "$(date -u '+%F %H:%M UTC') bootstrap $split"
  python -W ignore $OUT/cluster_bootstrap.py --split $split --out $OUT/bootstrap_$split.json || { echo "FAILED $split"; exit 1; }
done
echo "$(date -u '+%F %H:%M UTC') BOOTSTRAP DONE"
