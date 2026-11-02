"""Quick checks on geometry, radar features, fusion cells and metrics. Run: python -m clocs_r.test_core"""
import numpy as np
import torch

from clocs_r import aimotive, fusion, metrics, radar_features as rf


def box(x, y, yaw=0.0, l=4.0, w=2.0, h=1.5, z=0.75):
    return [x, y, z, l, w, h, yaw]


def main():
    a = np.array([box(50, 0)], dtype=np.float32)
    assert abs(metrics.iou_bev(a, a)[0, 0] - 1) < 1e-5 and abs(metrics.iou_3d(a, a)[0, 0] - 1) < 1e-5
    # 1 m longitudinal shift: overlap 3 of 4 m -> IoU 3/5
    b = np.array([box(51, 0)], dtype=np.float32)
    assert abs(metrics.iou_bev(a, b)[0, 0] - 0.6) < 1e-4
    # 90 degree rotation about the same center: overlap 2x2 of 8+8-4
    c = np.array([box(50, 0, np.pi / 2)], dtype=np.float32)
    assert abs(metrics.iou_bev(a, c)[0, 0] - 4 / 12) < 1e-4
    # half the height overlapping: 3D IoU 1/3
    d = np.array([box(50, 0, z=1.5)], dtype=np.float32)
    assert abs(metrics.iou_3d(a, d)[0, 0] - 1 / 3) < 1e-4
    # 45 degree: octagon area of two rectangles, checked against shapely-free analytic square case
    s1 = np.array([box(0, 0, 0, 2, 2)], dtype=np.float32)
    s2 = np.array([box(0, 0, np.pi / 4, 2, 2)], dtype=np.float32)
    octagon = 8 * (np.sqrt(2) - 1)
    assert abs(metrics.iou_bev(s1, s2)[0, 0] - octagon / (8 - octagon)) < 1e-4

    # NMS keeps the best of two overlapping boxes and the separate one
    boxes = np.array([box(10, 0), box(10.2, 0), box(30, 0)], dtype=np.float32)
    assert list(metrics.nms_bev(boxes, np.array([0.9, 0.8, 0.7]))) == [0, 2]

    # Radar features: returns on the first box only
    cand = np.array([box(100, 0), box(150, 10)], dtype=np.float32)
    pts = np.array([[100.5, 0.2, 0], [99.5, -0.2, 0], [100, 0, 0]], dtype=np.float32)
    vel = np.array([10, 12, 11], dtype=np.float32)
    f = rf.candidate_features(cand, pts, vel)
    assert abs(f[0, 0] - np.log(4)) < 1e-5 and abs(f[0, 1] - 11 / rf.V_MAX) < 1e-5
    assert list(f[1]) == [0, -1, -1, -1, -1]
    assert f[0, 4] < 1e-5  # one return exactly at the box center
    # Ego velocity: 2 m per frame at 0.1 s per frame is 20 m/s, also for a frame whose neighbors have poses but no timestamps
    seq = object.__new__(aimotive.Sequence)
    seq._period = None
    poses = {i: np.eye(4) for i in range(6)}
    for i in poses:
        poses[i][0, 3] = 2.0 * i
    seq.meta = dict(poses=poses, times={i: 0.1 * i for i in range(6) if i not in (2, 3)})
    assert abs(seq.ego_velocity(1)[0] - 20) < 1e-6 and abs(seq.ego_velocity(3)[0] - 20) < 1e-6
    # Computing a subset of the features gives the same values as computing all five
    for names in ([("r",), ("v",), ("r", "v"), ("r", "v", "e"), ("r", "delta", "o")]):
        part = rf.candidate_features(cand, pts, vel, names=names)
        for k, name in enumerate(rf.FEATURES):
            assert np.array_equal(part[:, k], f[:, k]) or name not in names
    # Table II channel counts
    channels = {k: 4 + len(v) for k, v in rf.FEATURE_SETS.items()}
    assert channels == dict(R0=4, R1=5, R2=5, R3=6, R4=7, R5=7, R6=7, R7=9)
    # Paper's stated parameter counts
    n_params = lambda m: sum(p.numel() for p in fusion.FusionNet(m).parameters())
    assert n_params(0) == 2143 and n_params(1) == 2161 and n_params(5) == 2233

    # Cells: candidate 0 overlaps two 2D boxes, candidate 1 none -> reserved row carries radar too
    b2d = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
    proj = np.array([[6, 6, 9, 9], [100, 100, 110, 110]], dtype=np.float32)
    radar = np.array([[0.5], [0.9]], dtype=np.float32)
    cells, col = fusion.build_cells(b2d, np.array([0.8, 0.6]), proj, np.array([0.3, 0.2]), np.array([0.1, 0.5]), radar)
    assert sorted(col.tolist()) == [0, 0, 1]
    row = cells[col == 1][0]
    assert row[0] == -1 and row[1] == -1 and row[2] == np.float32(0.2) and row[4] == np.float32(0.9)
    # Camera removed: every candidate is in the reserved row
    cells2, col2 = fusion.build_cells(np.zeros((0, 4)), np.zeros(0), proj, np.array([0.3, 0.2]), np.array([0.1, 0.5]), radar)
    assert sorted(col2.tolist()) == [0, 1] and (cells2[:, :2] == -1).all()
    # Max over cells per candidate
    net = fusion.FusionNet(1)
    out = net(torch.from_numpy(cells), torch.from_numpy(col), 2)
    with torch.no_grad():
        per = net.net(torch.from_numpy(cells).t()[None, :, None, :])[0, 0, 0]
    assert abs(out[0] - per[torch.from_numpy(col) == 0].max()) < 1e-6
    # The same at full frame size, on every device: small inputs hide unordered writes to repeated indices
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        x, column = torch.randn(40000, device=device), torch.randint(0, 2000, (40000,), device=device)
        expected = np.full(2000, -1e6, dtype=np.float32)
        np.maximum.at(expected, column.cpu().numpy(), x.cpu().numpy())
        assert np.array_equal(fusion._max_by_column(x, column, 2000).cpu().numpy(), expected)

    # Residual mode starts exactly at the 3D score, and takes the same cells as the plain network
    residual = fusion.FusionNet(1, residual=True)
    start = torch.sigmoid(residual(torch.from_numpy(cells), torch.from_numpy(col), 2))
    assert torch.allclose(start, torch.tensor([0.3, 0.2]), atol=1e-5)
    assert fusion.FusionNet(1, logit_input=True)(torch.from_numpy(cells), torch.from_numpy(col), 2).shape == (2,)
    # Doppler motion: a target closing at 15 m/s, seen 2 frames ago, moves 2 frames of closing nearer along its line of sight
    seq = object.__new__(aimotive.Sequence)
    seq._period, seq._radar = 0.1, dict(index={8: 0, 10: 1}, offsets=np.array([0, 1, 2]), rcs=np.array([5.0, 6.0]), frames=np.array([8, 10]),
                                        range=np.array([100.0, 97.0]), azimuth=np.zeros(2), elevation=np.zeros(2), speed=np.array([-15.0, -15.0]))
    seq.body_from_radar = np.eye(4)
    seq.meta = dict(poses={i: np.eye(4) for i in range(7, 12)}, times={i: 0.1 * i for i in range(7, 12)})
    pts, vel, rcs = seq.radar_accumulated_rcs(10, 3, motion=True)
    assert abs(pts[0, 0] - 97) < 1e-4 and abs(pts[1, 0] - 97) < 1e-4 and list(rcs) == [6.0, 5.0]
    assert abs(seq.radar_accumulated(10, 3)[0][1, 0] - 100) < 1e-4

    # Metrics: a perfect detection and a 4 m miss, in different bins
    gt = np.array([box(20, 0), box(120, 0)], dtype=np.float32)
    det = np.array([box(20, 0), box(124.5, 0), box(40, 0)], dtype=np.float32)
    r = metrics.evaluate([(gt, det, np.array([0.9, 0.8, 0.7]))])
    assert r["0-30"]["AP3D@0.5"] == 1.0 and r["100-200"]["AP3D@0.5"] == 0.0
    assert r["100-200"]["APCD@0.5"] == 0.0 and r["0-30"]["n_gt"] == 1 and r["30-50"]["n_gt"] == 0
    # Within 4 m is a CD hit only for the 4.0 threshold... 4.5 m is not
    assert r["100-200"]["APCD@4.0"] == 0.0
    r2 = metrics.evaluate([(gt[1:], np.array([box(123.5, 0)], dtype=np.float32), np.array([0.5]))])
    assert r2["100-200"]["APCD@4.0"] == 1.0 and r2["100-200"]["APCD@2.0"] == 0.0
    print("all core checks passed")


if __name__ == "__main__":
    main()
