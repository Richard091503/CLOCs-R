#!/usr/bin/env bash
# Phase 5 of results/next_round/PLAN.md: the ceiling of a radar detector. Injects the ground-truth boxes of the cars
# beyond 50 m with no LiDAR point and a radar return, and scores every saved model (seeds 1-6) on select and val.
# Models already in the output are skipped. From the repo root:
#   nohup setsid bash results/next_round/phase5_run.sh >> runs/next_round/phase5/chain.log 2>&1 &
cd "$(dirname "$0")/../.."
source env.sh
set -o pipefail
mkdir -p runs/next_round/phase5 results/next_round/phase5
stamp() { echo "== $(date '+%F %T') $*"; }
stamp "PHASE 5 START ($(df -h ~ | tail -1 | awk '{print $4}') free)"
python -W ignore -m clocs_r.oracle_inject --splits select,val --seeds 1,2,3,4,5,6 --out results/next_round/phase5/ceiling.json \
  >> runs/next_round/phase5/oracle.log 2>&1 || { stamp "FAILED: oracle_inject"; exit 1; }
stamp "PHASE 5 DONE ($(df -h ~ | tail -1 | awk '{print $4}') free)"
