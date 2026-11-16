# Final fusion networks

These are the final CLOCs-R (`R2+T2`) and CLOCs (`R0+T2`) fusion networks, seeds 1–6. They were trained with far weight 2 and the radar gate δ0 = 0.5 m, κ = 0.02. They are copies of `runs/ablation/sets_d0.5_k0.02/`, and `SHA256SUMS` lists their hashes.

To score them, copy them back and run the scoring step:

```bash
mkdir -p runs/ablation/sets_d0.5_k0.02 && cp results/full_dataset/weights/*.pth runs/ablation/sets_d0.5_k0.02/
python -m clocs_r.tune_fusion --configs R2+T2,R0+T2 --from_weights --weights runs/ablation/sets_d0.5_k0.02 \
    --far_weight 2 --gate 0.5,0.02 --seeds 1,2,3,4,5,6 --splits holdout,val --out /tmp/check.json
```

Scoring needs the candidates exported by SECOND and Cascade R-CNN. Their weights are too large for git (61 MB and 528 MB) and are not included here (see [`../METHODS.md`](../METHODS.md)).

**License:** these weights are derived from the aiMotive multimodal dataset, so they fall under its CC BY-NC-SA 4.0 license (non-commercial, share-alike).
