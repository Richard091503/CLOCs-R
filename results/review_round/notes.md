## What did not go as planned, and what to know

- **Caches rebuilt.** The base CLOCs cells (`runs/next_round/phase1/cells_in/{fit,select,val}_base.pkl`) had been
  deleted in a disk cleanup earlier on 2026-11-23, with the user's approval. They were rebuilt with `cache_cells`
  before tasks 1 and 5. A test rebuild of `select` that morning was byte-identical to the deleted file, and all three
  files came back at exactly their stored byte sizes. The reproduction checks of tasks 1 and 5 confirm the rebuilt
  cells give the stored results.
- **New cache files, none changed.**
  - The timing script's cell check wrote one new pair cache, `cells_in/val_rp_lcr_d2_dets_min0.05_m2.0.pkl` (12 MB). It
    is byte-identical to the existing `val_rp_phase7_d2_dets_min0.05_m2.0.pkl`, under another name.
  - Task 5's pair caches went to their own folders (`runs/review_round/cells_r*`).
  - No existing result, run output or cache was modified.
- **Disk.**
  - Free space reached 5.0 GB during the run. The cause was not this work: GNOME's file indexer
    (`tracker-miner-fs-3`) fell into an error loop over editor temporary files and wrote about 4.7 GB of warnings
    to `/var/log/syslog` and the system journal.
  - The indexer was stopped (`systemctl --user stop tracker-miner-fs-3`), and the growth stopped.
  - The user then masked the indexer and cleared the logs (`journalctl --vacuum-size=200M`, `truncate` of `syslog`),
    bringing free space to 9.3 GB before the timing run.
- **Parallel bootstrap.** Task 1's val bootstrap ran in parallel with `select`, in its own process, to save time.
  The results do not depend on this.
- **CPU-only steps.** In task 3, BEV NMS (`clocs_r.metrics.nms_bev`) is plain NumPy on the CPU. It computes rotated
  IoU between up to 1,000 boxes, so it dominates the post-detector time. A GPU NMS would be much faster, but none was
  used in any reported result.
