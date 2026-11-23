#!/usr/bin/env bash
# LCR end to end: detectors on fit, their candidates, the fusion networks (LC, LRd, LCR, LCR-D), val once, the bootstrap.
# Every step skips work whose output exists, so the script can be rerun after an interruption. From the repository root:
#   nohup setsid bash clocs_r/run_lcr.sh >> runs/lcr/run.log 2>&1 &
# Needs the environment (clocs_r/install_detectors.sh) and the dataset (clocs_r/fetch_aimotive.py, $AIMOTIVE_ROOT).
# Most of the time is detector training (SECOND, Cascade R-CNN about 2.5 h, D2 about 2 h on a 10 GB A100 slice).
# Disk: about 11 GB under runs/lcr, 8 GB of it the cached cells. D2's config reads the clutter classifier from
# runs/lcr/clutter.pt; with another LCR_RUNS, change CLUTTER in clocs_r/configs/pointpillar_radar_d2_aimotive.yaml.
cd "$(dirname "$0")/.."
source env.sh
set -o pipefail
R=${LCR_RUNS:-runs/lcr}
O=${LCR_RESULTS:-results/lcr/rerun}
D2=clocs_r/configs/pointpillar_radar_d2_aimotive.yaml
SPLITS=select,val,fit
mkdir -p "$R/det/second" "$R/det/cascade" "$R/cand/3d" "$R/cand/2d" "$R/cells" "$R/fusion" "$O"
stamp() { echo "== $(date '+%F %T') $*"; }
fail() { stamp "FAILED: $*"; exit 1; }
disk_ok() { local free; free=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ "$free" -ge 10 ] || fail "only ${free} GB free before: $1"; }
# Some CUDA start-ups failed on a shared GPU; a failed fusion step is retried
retry() { for a in 1 2 3; do "$@" && return 0; stamp "attempt $a failed: retrying in 60 s"; sleep 60; done; return 1; }
strip() { python -c "import torch,sys; c=torch.load(sys.argv[1],map_location='cpu'); c.pop('optimizer',None); torch.save(c,sys.argv[1])" "$1"; }
stamp "LCR START ($(df -h ~ | tail -1 | awk '{print $4}') free)"

# 1. LiDAR and camera detectors on fit: SECOND (40 epochs, range-weighted head) and Cascade R-CNN (4 epochs)
grep -q "training finished" "$R/det/second/train.log" 2>/dev/null || { stamp "SECOND"
  python -W ignore -m clocs_r.train_second --out "$R/det/second" --workers 6 --train_split fit --keep_every 10 --resume \
    > "$R/det/second/console.log" 2>&1 || fail "SECOND"; }
[ -f "$R/det/cascade/epoch_4.pth" ] || { stamp "Cascade R-CNN"
  python -W ignore -m clocs_r.cascade_2d train --out "$R/det/cascade" --epochs 4 --accum 2 --train_split fit \
    > "$R/det/cascade/console.log" 2>&1 && strip "$R/det/cascade/epoch_4.pth" || fail "Cascade"; }

# 2. Their candidates on every split, and each detector alone on select and val
disk_ok "candidates"
[ -f "$R/cand/3d/.done" ] || { stamp "SECOND candidates"; python -W ignore -m clocs_r.export_3d --ckpt "$R/det/second/checkpoint_epoch_40.pth" \
  --out "$R/cand/3d" --splits $SPLITS > "$R/export3d.log" 2>&1 && touch "$R/cand/3d/.done" || fail "export 3d"; }
[ -f "$R/cand/2d/.done" ] || { stamp "Cascade candidates"; python -W ignore -m clocs_r.cascade_2d export --out "$R/det/cascade" \
  --ckpt "$R/det/cascade/epoch_4.pth" --candidates "$R/cand/2d" --splits $SPLITS > "$R/export2d.log" 2>&1 && touch "$R/cand/2d/.done" || fail "export 2d"; }
[ -f "$O/detectors/aimotive_second.json" ] || python -W ignore -m clocs_r.eval_detectors --splits select,val --out "$O/detectors" \
  --cand3d "$R/cand/3d" --cand2d "$R/cand/2d" > "$R/eval_detectors.log" 2>&1 || fail "detectors alone"

# 3. CLOCs cells of every split (SECOND candidate x Cascade box), and the K40 floors
disk_ok "cells"
[ -f "$R/cells/fit_base.pkl" ] || { stamp "cells"; python -W ignore -m clocs_r.cache_cells --split $SPLITS --out "$R/cells" \
  --cand3d "$R/cand/3d" --cand2d "$R/cand/2d" > "$R/cells.log" 2>&1 || fail "cells"; }
python -W ignore -m clocs_r.visibility --split $SPLITS --cells "$R/cells" --cand3d "$R/cand/3d" --cand2d "$R/cand/2d" \
  > "$R/visibility.log" 2>&1 || fail "K40 floors"

# 4. Radar detector D2: the clutter classifier, then radar-only PointPillars (40 epochs), its boxes, and D2 alone
[ -f "$R/clutter.pt" ] || { stamp "clutter classifier"; python -W ignore -m clocs_r.radar_clutter --out "$R/clutter.pt" \
  --report "$O/clutter.json" > "$R/clutter.log" 2>&1 || fail "clutter"; }
[ -f "$R/d2/checkpoint_epoch_40.pth" ] || { stamp "D2"; python -W ignore -m clocs_r.train_second --config $D2 --out "$R/d2" \
  --train_split fit --workers 6 --keep_every 40 --resume 2>&1 | grep -v Warning > "$R/d2_train.log" || fail "D2"; }
[ -f "$R/d2_dets/DONE" ] || { stamp "D2 boxes"; python -W ignore -m clocs_r.export_3d --config $D2 --ckpt "$R/d2/checkpoint_epoch_40.pth" \
  --out "$R/d2_dets" --splits $SPLITS --nms --batch 16 2>&1 | grep -v Warning > "$R/d2_export.log" && touch "$R/d2_dets/DONE" || fail "D2 boxes"; }
[ -f "$O/detector_d2.json" ] || python -W ignore -m clocs_r.radar_candidates --detector select,val --cells "$R/cells" --cand3d "$R/cand/3d" \
  --cand2d "$R/cand/2d" --dets "$R/d2_dets" --tag d2 --out "$O/detector_d2.json" > "$R/d2_alone.log" 2>&1 || fail "D2 alone"

# 5. The fusion networks: 6 seeds each on fit, scored on select; then val once, from the saved networks
F="--cells $R/cells --cand3d $R/cand/3d --cand2d $R/cand/2d --radar_dets $R/d2_dets --weights $R/fusion --train_split fit"
for c in R0+T2+K40 R0+T2+RP R0+T2+K40+RC,R0+T2+K40+RC+DF; do
  for s in 1,2,3 4,5,6; do
    retry python -W ignore -m clocs_r.tune_fusion $F --configs $c --splits select --seeds $s --out "$O/fusion_select.json" \
      >> "$R/fusion.log" 2>&1 || fail "train $c $s"
    retry python -W ignore -m clocs_r.tune_fusion $F --configs $c --splits val --seeds $s --from_weights --out "$O/fusion_val.json" \
      >> "$R/fusion.log" 2>&1 || fail "val $c $s"
  done
  stamp "fusion $c"
done

# 6. Scene bootstrap, 1,000 resamples, on select and val
MODELS='[["L","second","",""],["LC","fusion","R0+T2+K40","'$R'/fusion"],["LRd","fusion","R0+T2+RP","'$R'/fusion"],
["LCR","fusion","R0+T2+K40+RC","'$R'/fusion"],["LCRD","fusion","R0+T2+K40+RC+DF","'$R'/fusion"]]'
COMP='[["LCR","L"],["LCR","LC"],["LCR","LRd"],["LCRD","LCR"],["LCRD","L"],["LRd","L"],["LC","L"]]'
for split in select val; do
  [ -f "$O/bootstrap_$split.json" ] || retry python -W ignore -m clocs_r.bootstrap --split $split --models "$MODELS" --comparisons "$COMP" \
    --cells "$R/cells" --cand3d "$R/cand/3d" --cand2d "$R/cand/2d" --radar_dets "$R/d2_dets" --out "$O/bootstrap_$split.json" \
    >> "$R/bootstrap.log" 2>&1 || fail "bootstrap $split"
done
stamp "LCR DONE ($(df -h ~ | tail -1 | awk '{print $4}') free)"
