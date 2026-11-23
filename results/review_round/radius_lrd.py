"""Review round, task 5 (post-hoc sensitivity check, not a selection): LRd (R0+T2+RP) with another radar pairing radius.

Runs clocs_r.tune_fusion unchanged, after setting radar_candidates.MATCH_M (default 2.0 m, left unchanged in the code)
to --radius before radar_pairs reads it; affinity = 1 - d / radius. The radius is recorded in each saved network's
settings (cfg["rp"]["match_m"]). Every other argument goes to tune_fusion as given.

    cd ~/Documents/CLOCs-R && source env.sh
    python ../CLOCs-R-review/results/review_round/radius_lrd.py --radius 1.0 --configs R0+T2+RP --seeds 1,2,3 --splits select \\
        --cells runs/review_round/cells_r1.0 --weights runs/review_round/fusion_r1.0 --out ../CLOCs-R-review/results/review_round/radius/lrd_r1.0.json
"""
import sys

if __name__ == "__main__":
    i = sys.argv.index("--radius")
    radius = float(sys.argv[i + 1])
    del sys.argv[i:i + 2]
    from clocs_r import radar_candidates
    radar_candidates.MATCH_M = radius
    from clocs_r import radar_pairs, tune_fusion
    assert radar_pairs.MATCH_M == radius and radar_pairs.frame_pairs.__defaults__[1] == radius
    assert tune_fusion.parse("R0+T2+RP")["rp"]["match_m"] == radius
    print(f"radar pairing radius {radius} m", flush=True)
    tune_fusion.main()
