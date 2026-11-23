#!/bin/bash
# Review round, task 3: timing of every LCR step on 200 val frames, with nothing else running. Records the hardware first.
#   nohup setsid bash ~/Documents/CLOCs-R-review/results/review_round/run_timing.sh >> ~/Documents/CLOCs-R/runs/review_round/timing.log 2>&1 &
cd ~/Documents/CLOCs-R || exit 1
source env.sh
OUT=../CLOCs-R-review/results/review_round
python - <<'PY' > $OUT/hardware.json
import json, os, subprocess, torch
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
print(json.dumps(dict(gpu=f"{gpu.split(',')[0]} (an NVIDIA vGPU slice of an A100: {gpu.split(',')[1].strip()}, driver {gpu.split(',')[2].strip()})",
                      cpu=cpu, cores=os.cpu_count(),
                      torch=f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, {torch.get_num_threads()} CPU threads"), indent=1))
PY
for p in params second d2 lcr cascade; do
  echo "$(date -u '+%F %H:%M UTC') timing $p"
  python -W ignore $OUT/timing.py --part $p --n 200 --out $OUT/timing_$p.json > /dev/null 2>> runs/review_round/timing_err.log || { echo "FAILED $p"; exit 1; }
done
rm -rf $OUT/_cascade_cfg
echo "$(date -u '+%F %H:%M UTC') TIMING DONE"
