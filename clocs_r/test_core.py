"""Quick checks on geometry, radar features, fusion cells and metrics. Run: python -m clocs_r.test_core"""
import os

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

    # Range switch: fused scores under 50 m, SECOND's from 50 m on (the boundary goes to SECOND)
    from clocs_r import range_switch
    sw = range_switch.switch_scores(np.array([box(30, 0), box(50, 0), box(40, 40)], np.float32),
                                    np.array([0.9, 0.9, 0.9]), np.array([0.1, 0.2, 0.3]), 50.0)
    assert list(sw) == [0.9, 0.2, 0.3]

    # Phase 3 pairing filter: candidate 0 has pairs of IoU 0.05 and 0.3, candidate 1 only the no-camera row
    from clocs_r import train_fusion, tune_fusion, visibility
    column = np.array([0, 0, 1], np.int16)
    pairs = np.array([[0.05, 0.9], [0.3, 0.5], [-1, -1]], np.float16)
    c, p = fusion.filter_pairs(column, pairs, 2)
    assert np.array_equal(c, column) and np.array_equal(p, pairs)  # defaults keep everything, in the same order
    c, p = fusion.filter_pairs(column, pairs, 2, min_iou=0.1)
    assert c.tolist() == [0, 1] and p[0].tolist() == [np.float16(0.3), np.float16(0.5)] and p[1].tolist() == [-1, -1]
    c, p = fusion.filter_pairs(column, pairs, 2, s2d_floor=0.6)  # only the 0.9 pair survives a floor of 0.6
    assert c.tolist() == [0, 1] and p[0, 1] == np.float16(0.9)
    c, p = fusion.filter_pairs(column, pairs, 2, min_iou=0.5)  # nothing left: both candidates in the reserved row
    assert sorted(c.tolist()) == [0, 1] and (p == -1).all()
    # frame_cells: extra channels follow the radar ones, per candidate; pairing is applied before them
    item = dict(column=column, pairs=pairs, scores=np.array([0.4, 0.6], np.float32), dist=np.array([0.1, 0.2], np.float32))
    rad = np.array([[0.5, 7], [0.9, 8]], np.float32)
    extra = np.array([[0.25, 0.75], [0.125, 0.5]], np.float32)
    x, col = train_fusion.frame_cells(item, rad, [0], "cpu", extra=extra, pairing=(0.1, -1.0))
    assert x.shape == (2, 7) and col.tolist() == [0, 1]
    assert x[0].tolist() == [np.float32(np.float16(0.3)), 0.5, np.float32(0.4), np.float32(0.1), 0.5, 0.25, 0.75]
    assert x[1, 4:].tolist() == [np.float32(0.9), 0.125, 0.5]
    x0, col0 = train_fusion.frame_cells(item, rad, [0], "cpu")
    assert x0.shape == (3, 5) and col0.tolist() == [0, 0, 1]
    assert fusion.FusionNet(1 + 2)(x, col, 2).shape == (2,)
    # Points inside rotated boxes: a 4 x 2 m box turned 90 degrees holds (0, 1.9) but not (1.9, 0); height is checked too
    pts = np.array([[0, 1.9, 0.5], [1.9, 0, 0.5], [0, 0, 3.0], [0.5, -1.5, 1.0], [30, 0, 0.5]], np.float32)
    assert visibility.points_in_boxes(np.array([box(0, 0, np.pi / 2), box(30, 0)], np.float32), pts).tolist() == [2, 1]
    # Projected height: a camera looking along body x with focal 1000 sees a 1.5 m tall box whose near face is 98 m away
    seq = object.__new__(aimotive.Sequence)
    seq.cam_from_body = np.array([[0, -1, 0, 0], [0, 0, -1, 1.0], [1, 0, 0, 0], [0, 0, 0, 1]], float)
    seq.meta = dict(dist=np.zeros(5), focal=(1000.0, 1000.0), principal=(640.0, 352.0), image_size=(1280, 704))
    h, valid = visibility.projected_height(seq, np.array([box(100, 0), box(-20, 0)], np.float32))
    assert abs(h[0] - 1000 * 1.5 / 98) < 1e-3 and valid.tolist() == [True, False]
    f = visibility.features(seq, np.array([box(100, 0), box(-20, 0)], np.float32), np.zeros((0, 3), np.float32))
    assert abs(f[0, 0] - np.log1p(1500 / 98) / np.log1p(704)) < 1e-5 and f[1, 0] == fusion.MISSING and (f[:, 1] == 0).all()
    assert visibility.top_score(np.arange(50.0)) == 10.0 and visibility.top_score(np.arange(40.0)) == fusion.MISSING
    # Old configuration names parse exactly as before (saved weights are checked against these settings)
    assert "p3" not in tune_fusion.parse("R2+T2") and "p3" not in tune_fusion.parse("R0+T2")
    cfg = tune_fusion.parse("R2+T2+P+H+I2+D3+S100")
    assert cfg["p3"]["vis"] == ["height", "points"] and cfg["p3"]["min_iou"] == 0.25 and cfg["p3"]["cam_dropout"] == 0.3
    assert cfg["p3"]["oversample_m"] == 100.0 and cfg["p3"]["top2d"] == 0 and tune_fusion.n_extra(cfg) == 2
    try:
        tune_fusion.parse("R2+T2+I1+K40")
        raise RuntimeError("two pairing alternatives were accepted")
    except AssertionError:
        pass

    # Phase 4 gated model: starts exactly at SECOND's logit, cam_term is the max over cells, radar and context per candidate
    torch.manual_seed(0)
    g = fusion.GatedFusion(num_radar=1, num_context=4)
    cells = torch.tensor([[0.3, 0.8, 0.4, 0.1, 0.5, 0.2, 0.3, 0.0, 0.1],
                          [0.6, 0.5, 0.4, 0.1, 0.5, 0.2, 0.3, 0.0, 0.1],
                          [-1, -1, 0.7, 0.4, -1, -1, 0.1, 0.0, 0.0]])
    col = torch.tensor([0, 0, 1])
    assert torch.allclose(g(cells, col, 2), torch.logit(torch.tensor([0.4, 0.7])), atol=1e-5)
    assert torch.allclose(g.last_gates, torch.sigmoid(g.f_gate(torch.tensor([[0.2, 0.3, 0.0, 0.1, 0.1], [-1, 0.1, 0.0, 0.0, 0.4]]))))
    for f in (g.f_cam, g.f_radar):
        torch.nn.init.normal_(f[-1].weight)
        torch.nn.init.constant_(f[-1].bias, 0.1)
    with torch.no_grad():
        out = g(cells, col, 2)
        cam = g.f_cam(cells[:, :4])[:, 0]
        rad = g.f_radar(torch.tensor([[0.5, 0.1, 0.4], [-1, 0.4, 0.7]]))[:, 0]
        gates = g.last_gates
        want = torch.logit(torch.tensor([0.4, 0.7])) + gates[:, 0] * torch.stack([cam[:2].max(), cam[2]]) + gates[:, 1] * rad
    assert torch.allclose(out, want, atol=1e-5)
    assert fusion.GatedFusion(0, 2).f_radar is None and fusion.GatedFusion(0, 2)(cells[:, :6], col, 2).shape == (2,)
    assert 2000 < sum(p.numel() for p in fusion.GatedFusion(1, 4).parameters()) < 3000
    # Moving returns: only returns of at least 1 m/s inside the gate count
    m = visibility.moving_returns(np.array([box(50, 0)], np.float32), np.array([[50, 0, 0], [50.5, 0, 0], [80, 0, 0]], np.float32),
                                  np.array([5.0, 0.2, 9.0], np.float32))
    assert abs(m[0] - np.log1p(1)) < 1e-6
    # Gated configuration: context of height, points, radar returns, moving returns, condition one-hot
    cfg = tune_fusion.parse("R2+T2+K40+G+C")
    assert cfg["gated"]["condition"] and tune_fusion.context_size(cfg) == 8 and tune_fusion.context_size(tune_fusion.parse("R0+T2+G")) == 2
    assert "gated" not in tune_fusion.parse("R2+T2+K40")
    tune_fusion.DATA.update(vis_t=dict(features=[np.array([[0.4, 0.5], [0.3, 0.2]], np.float16)]), moving_t=[np.array([0.7, 0.0], np.float16)])
    ctx = tune_fusion.gate_context(cfg, "t", [dict(scores=np.zeros(2), odd="night")], [np.array([[1.1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], np.float16)])
    assert ctx[0].shape == (2, 8) and ctx[0][0, :4].tolist() == [np.float16(0.4), np.float16(0.5), np.float16(1.1), np.float16(0.7)]
    assert ctx[0][:, 4:].tolist() == [[0, 0, 1, 0]] * 2 and tune_fusion.splits.ODDS[2] == "night"

    # Phase 5 injection: targets, SECOND's finds, and appended candidates that leave the others' scores unchanged
    from clocs_r import oracle_inject as oi
    gt = np.array([box(30, 0), box(60, 0), box(120, 3), box(150, -3)], np.float32)
    assert oi.targets(gt, np.array([0, 0, 2, 0]), np.array([1.0, 0.7, 1.0, 0.0])).tolist() == [False, True, False, False]
    assert oi.found_by(gt, np.array([box(61, 0.5), box(120, 3)], np.float32), np.array([0.3, 0.01])).tolist() == [False, True, False, False]
    item = dict(seq="s", frame=0, boxes=np.array([box(20, 0), box(60, 1)], np.float32), scores=np.array([0.8, 0.2], np.float32),
                dist=np.array([0.1, 0.3], np.float32), keep_eval=np.array([True, True]), target=np.zeros(2, np.float32),
                weight=np.ones(2, np.float32), pairs=np.array([[0.5, 0.9], [-1, -1]], np.float16), column=np.array([0, 1], np.int16))
    inj = [dict(key=("s", 0), inject=dict(boxes=np.array([box(70, 2)], np.float32), pairs=np.array([[0.3, 0.6]], np.float16),
                                          column=np.array([0], np.int16), dist=np.array([0.35], np.float32),
                                          radar=np.array([[0.7, 0.2, -1, -1, -1]], np.float16), vis=np.array([[0.2, 0.0]], np.float16),
                                          moving=np.array([0.7], np.float16), keep_eval=np.array([True])))]
    rad = [np.array([[1.0, 0.1, 0, 0, 0], [0, -1, -1, -1, -1]], np.float16)]
    vis = dict(features=[np.array([[0.4, 0.5], [0.3, 0.2]], np.float16)], top2d=np.array([-1.0], np.float32), keys=[("s", 0)])
    b2, r2, v2, m2 = oi.augment([item], rad, vis, [np.array([0.0, 0.7], np.float16)], inj)
    a = b2[0]
    assert a["scores"].tolist() == [np.float32(0.8), np.float32(0.2), np.float32(oi.S_LOW)] and a["injected"].tolist() == [False, False, True]
    assert a["column"].tolist() == [0, 1, 2] and len(a["boxes"]) == 3 and r2[0].shape == (3, 5) and v2["features"][0].shape == (3, 2)
    assert m2[0].tolist() == [0.0, np.float16(0.7), np.float16(0.7)] and len(item["scores"]) == 2  # the original item is unchanged
    cfg = tune_fusion.parse("R2+T2+K40+G")
    torch.manual_seed(0)
    g = tune_fusion.make_net(cfg, [1])
    for f in (g.f_cam, g.f_radar):
        torch.nn.init.normal_(f[-1].weight)
    outs = []
    for bb, rr, vv, mm in (([item], rad, vis, [np.array([0.0, 0.7], np.float16)]), (b2, r2, v2, m2)):
        tune_fusion.DATA.update(vis_t=vv, moving_t=mm)
        cx = tune_fusion.gate_context(cfg, "t", bb, rr)
        x, col = train_fusion.frame_cells(bb[0], rr[0], [1], "cpu", False, cx[0], (0.0, -1.0))
        outs.append(g(x, col, len(bb[0]["scores"])).detach())
    assert torch.allclose(outs[0], outs[1][:2]) and outs[1].shape == (3,)
    recs, kept = oi.variant_frames([dict(a, gt=np.zeros((0, 7), np.float32), odd="urban")], [np.array([0.9, 0.5, 0.01])], "perfect")
    assert kept == 1 and recs[0][2].max() == 1.0
    recs, kept = oi.variant_frames([dict(a, gt=np.zeros((0, 7), np.float32), odd="urban")], [np.array([0.9, 0.5, 0.01])], "none")
    assert kept == 0 and len(recs[0][1]) == 2

    # Phase 6 radar points: sweeps accumulated with the ego poses, age per sweep, Doppler and RCS scaled
    try:
        from clocs_r import radar_dataset
    except ImportError:  # pcdet is not installed
        radar_dataset = None
    if radar_dataset is not None:
        class FakeSeq:
            frame_period = 0.1
            meta = dict(poses={5: np.eye(4), 4: np.array([[1, 0, 0, -2.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])})
            radar = dict(index={5: 0, 4: 1}, offsets=[0, 1, 3], rcs=np.array([10.0, 1.0, 1000.0]))

            def radar_sweep(self, f):
                if f == 5:
                    return np.array([[50.0, 1, 0]], np.float32), np.array([8.0], np.float32)
                if f == 4:
                    return np.array([[60.0, 0, 0], [70, 0, 0]], np.float32), np.array([0.0, -4.0], np.float32)
                return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
        pts = radar_dataset.radar_points(FakeSeq(), 5, m=3)
        assert pts.shape == (3, 6) and pts[1:, 0].tolist() == [58.0, 68.0]  # the older sweep moved by the ego pose
        assert np.allclose(pts[:, 3], [0.2, 0.0, -0.1]) and np.allclose(pts[:, 5], [0.0, 0.1, 0.1])
        lo, hi = rf.RCS_LOG_RANGE
        assert np.allclose(pts[:, 4], [(1 - lo) / (hi - lo), (0 - lo) / (hi - lo), (3 - lo) / (hi - lo)])
        # The pillar scatter matches OpenPCDet's, and keeps a trailing frame with no pillar as an empty map
        from easydict import EasyDict
        from pcdet.models.backbones_2d.map_to_bev import PointPillarScatter
        sc_cfg = EasyDict(NUM_BEV_FEATURES=3)
        pf, co = torch.randn(4, 3), torch.tensor([[0, 0, 1, 2], [1, 0, 3, 4], [1, 0, 0, 0], [0, 0, 2, 1]])
        old = PointPillarScatter(sc_cfg, (5, 4, 1))(dict(pillar_features=pf, voxel_coords=co))["spatial_features"]
        new = radar_dataset.RadarPillarScatter(sc_cfg, (5, 4, 1))(dict(pillar_features=pf, voxel_coords=co, batch_size=3))["spatial_features"]
        assert torch.equal(old, new[:2]) and new.shape[0] == 3 and new[2].abs().sum() == 0
        # Phase 6b D2: Doppler motion correction moves an older return along its line of sight by v * age
        mpts = radar_dataset.radar_points(FakeSeq(), 5, m=3, motion=True)
        assert np.allclose(mpts[0, :3], pts[0, :3]) and np.allclose(mpts[1:, 0], [58.0, 68.0 - 4.0 * 0.1])
        from clocs_r import radar_clutter
        cl = radar_clutter.features(pts)
        assert cl.shape == (3, 12) and np.allclose(cl[:, :6], pts) and np.allclose(cl[:, 6], np.hypot(pts[:, 0], pts[:, 1]) / 200)
        assert cl[0, 7] == np.float32(np.log1p(1)) and np.allclose(cl[1:, 7], np.log1p(1))  # 58, 68 and 50: no neighbour within 2 m
        class BoxSeq:
            def boxes(self, frame, classes):
                return np.array([box(58.5, 0)], np.float32), ["Car"]
        assert radar_clutter.labels(BoxSeq(), 5, pts).tolist() == [0, 1, 0]  # 50 m is outside 58.5 +- (2.1 + 0.5)

    # Phase 6 radar candidates: rd for SECOND candidates, unmatched detections become candidates, distance targets
    from clocs_r import radar_candidates as rc
    sec_b = np.array([box(60, 0), box(100, 0), box(150, 5)], np.float32)
    rd, new = rc.match(sec_b, np.array([0.5, 0.01, 0.2]), np.array([box(61, 0.5), box(101, 0), box(180, 0), box(150, 5)], np.float32),
                       np.array([0.3, 0.6, 0.4, 0.02]))
    assert np.allclose(rd, [0.3, 0.6, 0.0]) and new.tolist() == [False, True, True, False]  # 101: only a 0.01 SECOND box nearby
    t, w = rc.targets(np.array([box(101, 0), box(180, 0), box(120, 3)], np.float32), np.array([box(100, 1), box(123, 3)], np.float32),
                      np.array([False, False, False]))
    assert t.tolist() == [1, 0, 0] and w.tolist() == [1, 1, 0]
    inj = dict(boxes=np.array([box(101, 0)], np.float32), scores=np.array([0.6], np.float32), pairs=np.array([[-1, -1]], np.float16),
               column=np.array([0], np.int16), dist=np.array([0.5], np.float32), radar=np.array([[0.7, 0.3, -1, -1, -1]], np.float16),
               vis=np.array([[-1, 0.0]], np.float16), moving=np.array([0.7], np.float16), keep_eval=np.array([True]),
               target=np.array([1.0], np.float32), weight=np.array([1.0], np.float32))
    b3, r3, v3, m3 = rc.augment([dict(item)], [rad[0].copy()], dict(vis, features=list(vis["features"])), [np.array([0.0, 0.7], np.float16)],
                                [dict(key=("s", 0), rd=np.array([0.0, 0.3], np.float16), inject=inj)])
    a = b3[0]
    assert a["src"].tolist() == [False, False, True] and a["scores"][2] == np.float32(0.6) and a["column"].tolist() == [0, 1, 2]
    assert r3[0].shape == (3, 7) and r3[0][:, 5].tolist() == [0, np.float16(0.3), np.float16(0.6)] and r3[0][:, 6].tolist() == [0, 0, 1]
    assert a["target"].tolist() == [0, 0, 1] and v3["features"][0].shape == (3, 2) and m3[0].shape == (3,)
    cfg = tune_fusion.parse("R2+T2+K40+G+RD")
    assert tune_fusion.names_idx(cfg) == [1, 5, 6] and tune_fusion.context_size(cfg) == 5 and "rd" not in tune_fusion.parse("R2+T2+K40+G")
    tune_fusion.DATA.update(vis_t=v3, moving_t=m3)
    cx = tune_fusion.gate_context(cfg, "t", b3, r3)
    assert cx[0].shape == (3, 5) and cx[0][:, 4].tolist() == [0, 0, 1]
    g = tune_fusion.make_net(cfg, tune_fusion.names_idx(cfg))
    x, col = train_fusion.frame_cells(a, r3[0], tune_fusion.names_idx(cfg), "cpu", False, cx[0], (0.0, -1.0))
    assert x.shape[1] == 4 + 3 + 5 and g(x, col, 3).shape == (3,)

    # Phase 7: NC (no camera) and RP (radar-detector boxes as CLOCs' pairs); old names parse unchanged
    for old in ("R0+T2", "R2+T2+K40+G", "R2+T2+K40+G+RD"):
        assert "nocam" not in tune_fusion.parse(old) and "rp" not in tune_fusion.parse(old)
    assert tune_fusion.parse("R0+T2+NC")["nocam"] and tune_fusion.names_idx(tune_fusion.parse("R0+T2+RP")) == []
    from clocs_r import radar_pairs
    cand, aff, sc = radar_pairs.frame_pairs(np.array([box(60, 0), box(100, 0), box(150, 0)], np.float32),
                                            np.array([box(60.5, 0), box(101.5, 0), box(150, 0)], np.float32), np.array([0.6, 0.3, 0.02], np.float32))
    assert cand.tolist() == [0, 1] and np.allclose(aff, [0.75, 0.25]) and np.allclose(sc, [0.6, 0.3])  # 0.02 < MIN_SCORE
    col, prs = radar_pairs.cells(3, cand, aff, sc)
    assert col.tolist() == [0, 1, 2] and prs[2].tolist() == [fusion.MISSING, fusion.MISSING] and abs(float(prs[0, 0]) - 0.75) < 1e-3
    rp_item = radar_pairs.replace_pairs([dict(item, scores=np.zeros(3, np.float32))], [(cand, aff, sc)])[0]
    assert rp_item["column"].tolist() == [0, 1, 2] and item["column"].tolist() == [0, 1]  # the original item is unchanged
    x, col = train_fusion.frame_cells(dict(rp_item, scores=np.array([0.5, 0.4, 0.3], np.float32), dist=np.array([0.3, 0.5, 0.7], np.float32)),
                                      None, [], "cpu")
    assert x.shape == (3, 4) and torch.allclose(x[:, 2], torch.tensor([0.5, 0.4, 0.3]))

    # Phase 7 LCR (token RC): K40 camera cells and radar-detector cells in one network, s3D kept in column 2
    cfg_rc = tune_fusion.parse("R0+T2+K40+RC")
    assert cfg_rc["rc"] and tune_fusion.n_extra(cfg_rc) == 2 and tune_fusion.names_idx(cfg_rc) == []
    rc_item = dict(scores=np.array([0.5, 0.4, 0.3, 0.2], np.float32), dist=np.array([0.1, 0.2, 0.3, 0.4], np.float32),
                   column=np.array([0, 0, 1, 3], np.int32), pairs=np.array([[0.5, 0.9], [0.3, 0.1], [0.6, 0.8], [-1, -1]], np.float16))
    rp_fr = (np.array([1, 2], np.int32), np.array([0.5, 0.25], np.float32), np.array([0.7, 0.2], np.float32))
    col_rc, pr_rc = radar_pairs.combined_cells(rc_item, rp_fr, 0.15)  # the floor 0.15 drops the (0.3, 0.1) camera pair
    assert col_rc.tolist() == [0, 1, 1, 2, 3]
    assert np.allclose(pr_rc.astype(np.float32), [[0.5, 0.9, -1, -1], [0.6, 0.8, -1, -1], [-1, -1, 0.5, 0.7], [-1, -1, 0.25, 0.2],
                                                  [-1, -1, -1, -1]], atol=1e-3)
    rc_it = radar_pairs.add_pairs([rc_item], [rp_fr], [0.15])[0]
    assert rc_item["pairs"].shape == (4, 2)  # the original item is unchanged
    x_rc, c_rc = train_fusion.frame_cells(rc_it, None, [], "cpu")
    assert x_rc.shape == (5, 6) and torch.allclose(x_rc[:, 2], torch.tensor([0.5, 0.4, 0.4, 0.3, 0.2]))
    assert torch.allclose(x_rc[:, 3], torch.tensor([0.1, 0.2, 0.2, 0.3, 0.4])) and torch.allclose(x_rc[:, 4:], torch.from_numpy(pr_rc[:, 2:].astype(np.float32)))
    net_rc = tune_fusion.make_net(cfg_rc, [])
    z_rc = net_rc(x_rc, c_rc, 4)  # the T2 residual starts at SECOND's logit
    assert torch.allclose(z_rc, torch.logit(torch.tensor([0.5, 0.4, 0.3, 0.2])), atol=1e-5)
    assert tune_fusion.parse("R0+T2+K40").get("rc") is None and tune_fusion.n_extra(tune_fusion.parse("R0+T2+K40")) == 0
    x_lc, _ = train_fusion.frame_cells(rc_item, None, [], "cpu", pairing=(0.0, 0.15))  # LC's layout is unchanged
    assert x_lc.shape[1] == 4 and torch.allclose(x_lc[0], torch.tensor([0.5, 0.9, 0.5, 0.1]), atol=1e-3)

    # Phase 7 LCR-D (token DF): far candidates lose their camera cells in training only; radar cells stay
    cfg_df = tune_fusion.parse("R0+T2+K40+RC+DF")
    assert cfg_df["rc"] and cfg_df["far_cam_dropout"] == dict(far_m=50.0, p=0.5) and tune_fusion.n_extra(cfg_df) == 2
    assert "far_cam_dropout" not in cfg_rc  # LCR's settings, and so its saved weights, are unchanged
    try:
        tune_fusion.parse("R0+T2+K40+DF")
        raise RuntimeError("DF without RC should fail")
    except AssertionError:
        pass
    far_df = np.array([False, True, True, False])  # candidate 1 has camera and radar cells, 2 radar only, 3 reserved
    rng_df = np.random.RandomState(0)
    c1, p1 = fusion.drop_far_camera(rc_it["column"], rc_it["pairs"], far_df, 1.0, rng_df)
    assert c1.tolist() == [0, 1, 2, 3] and np.allclose(p1.astype(np.float32), pr_rc[[0, 2, 3, 4]].astype(np.float32))
    c0, p0 = fusion.drop_far_camera(rc_it["column"], rc_it["pairs"], far_df, 0.0, rng_df)
    assert c0.tolist() == col_rc.tolist() and np.array_equal(p0, rc_it["pairs"])
    only_cam = dict(column=np.array([0, 1], np.int32), pairs=np.array([[0.5, 0.9, -1, -1], [0.6, 0.8, -1, -1]], np.float16))
    c2, p2 = fusion.drop_far_camera(only_cam["column"], only_cam["pairs"], np.array([False, True]), 1.0, rng_df)
    assert c2.tolist() == [0, 1] and np.allclose(p2[1].astype(np.float32), [-1, -1, -1, -1])  # left with the reserved cell
    assert rc_it["pairs"].shape == (5, 4) and only_cam["pairs"].shape == (2, 4)  # inputs unchanged
    hits = [len(fusion.drop_far_camera(rc_it["column"], rc_it["pairs"], far_df, 0.5, rng_df)[0]) for _ in range(2000)]
    assert abs(np.mean(np.array(hits) == 4) - 0.5) < 0.05  # candidate 1's camera cell is dropped about half the time
    df_train = [dict(rc_it, target=np.array([1, 0, 0, 0], np.float32), weight=np.ones(4, np.float32))]
    net_df, _ = train_fusion.fit(df_train, [None], [], 1, 1, "cpu", model=lambda: tune_fusion.make_net(cfg_df, []),
                                 far_cam_dropout=(0.15, 0.5))
    assert df_train[0]["pairs"].shape == (5, 4)  # training does not modify the cached cells

    # Phase 7 selector: starts near LiDAR alone, combines expert logits, camera evidence from K40 pairs
    from clocs_r import selector
    gsel = selector.Gate()
    lg = torch.tensor([[1.0, -2.0, 3.0], [0.0, 4.0, -4.0]])
    o = gsel(torch.zeros(2, selector.N_CONTEXT), lg)
    w0 = np.exp(2) / (np.exp(2) + 2)
    assert np.allclose(gsel.last_weights.numpy(), [[w0, (1 - w0) / 2, (1 - w0) / 2]] * 2, atol=1e-6)
    assert torch.allclose(o, (gsel.last_weights * lg).sum(1))
    import tempfile
    # Phase 7 blend ablation: the Gate's starting weights, the same for every candidate whatever the context
    bl = selector.Blend()
    ob = bl(torch.randn(2, selector.N_CONTEXT), lg)
    assert torch.allclose(bl.last_weights, gsel.last_weights, atol=1e-6) and torch.allclose(ob, o, atol=1e-6)
    assert sum(p.numel() for p in bl.parameters()) == 3
    bl.bias.data = torch.tensor([0.0, 1.0, -1.0])
    bl(torch.randn(2, selector.N_CONTEXT), lg)
    assert torch.allclose(bl.last_weights[0], bl.last_weights[1]) and torch.allclose(bl.last_weights[0], torch.softmax(bl.bias, 0))
    with tempfile.TemporaryDirectory() as td:
        torch.save(dict(model=bl.state_dict(), radar_expert="X", seed=1, init_bias=selector.INIT_BIAS, kind="blend"),
                   selector.gate_path(td, "X", 1, "blend"))
        assert torch.allclose(selector.load_gate(td, "X", 1, "cpu", "blend").bias, bl.bias)
        torch.save(dict(model=gsel.state_dict(), radar_expert="X", seed=1, init_bias=selector.INIT_BIAS),  # saved before kinds
                   selector.gate_path(td, "X", 1))
        assert isinstance(selector.load_gate(td, "X", 1, "cpu"), selector.Gate)
    cam_item =dict(scores=np.array([0.5, 0.4, 0.3], np.float32), column=np.array([0, 0, 1, 2], np.int16),
                    pairs=np.array([[0.5, 0.9], [0.8, 0.2], [0.6, 0.1], [-1, -1]], np.float16), dist=np.array([0.1, 0.3, 0.6], np.float32))
    ev = selector.camera_evidence(cam_item, 0.15)  # K40 floor 0.15 drops the 0.1 pair
    assert np.allclose(ev, [0.45, -1.0, -1.0], atol=1e-3)
    ctx7 = selector.context(cam_item, np.zeros((3, 3), np.float32), np.zeros((3, 2), np.float16), np.zeros((3, 5), np.float16),
                            np.zeros(3, np.float16), 0.15)
    assert ctx7.shape == (3, selector.N_CONTEXT) and np.allclose(ctx7[:, 0], cam_item["dist"]) and np.allclose(ctx7[:, -1], cam_item["scores"])

    # Phase 7 ceiling: the compiled NMS + APCD equals metrics.evaluate after metrics.nms_bev, overall and on a frame subset
    from clocs_r import selector_ceiling as scl
    rng_c = np.random.default_rng(5)
    items_c, sc_c = [], []
    for f in range(12):
        n, g = int(rng_c.integers(0, 60)), int(rng_c.integers(0, 8))
        ctr = rng_c.uniform(-150, 150, (max(g, 1), 2))
        gt = np.column_stack([ctr[:g], np.zeros(g), np.full((g, 3), [4.5, 1.9, 1.6]), rng_c.uniform(-3, 3, g)])
        src = ctr[rng_c.integers(0, len(ctr), n)] + rng_c.normal(0, 1.5, (n, 2))  # duplicates around objects, so NMS acts
        boxes = np.column_stack([src, np.zeros(n), np.full((n, 3), [4.5, 1.9, 1.6]), rng_c.uniform(-3, 3, n)]).astype(np.float32)
        items_c.append(dict(boxes=boxes, keep_eval=rng_c.random(n) < 0.9, gt=gt.astype(np.float32), odd="urban" if f % 2 else "night"))
        sc_c.append(rng_c.random(n).astype(np.float32))
    scorer = scl.Scorer(items_c, [it["odd"] for it in items_c])
    got = scorer.apcd(scorer.run(scorer.flat(sc_c)))
    ref = metrics.evaluate([(it["gt"], it["boxes"][it["keep_eval"]][k], s[it["keep_eval"]][k]) for it, s in zip(items_c, sc_c)
                            for k in [metrics.nms_bev(it["boxes"][it["keep_eval"]], s[it["keep_eval"]], 0.1)]], bins=scl.EVAL)
    for b in scl.NAMES:
        assert abs(got[b] - ref[b]["APCD"]) < 1e-9 or (np.isnan(got[b]) and np.isnan(ref[b]["APCD"])), (b, got[b], ref[b]["APCD"])
    night = np.array([it["odd"] == "night" for it in items_c])
    got_n = scorer.apcd(scorer.run(scorer.flat(sc_c)), night)
    ref_n = metrics.evaluate([(it["gt"], it["boxes"][it["keep_eval"]][k], s[it["keep_eval"]][k]) for it, s in zip(items_c, sc_c) if it["odd"] == "night"
                              for k in [metrics.nms_bev(it["boxes"][it["keep_eval"]], s[it["keep_eval"]], 0.1)]], bins=scl.EVAL)
    for b in scl.NAMES:
        assert abs(got_n[b] - ref_n[b]["APCD"]) < 1e-9 or (np.isnan(got_n[b]) and np.isnan(ref_n[b]["APCD"])), (b, got_n[b], ref_n[b]["APCD"])
    assert len(scl.simplex(0.05)) == 231 and np.allclose(scl.simplex(0.05).sum(1), 1)

    # Scene bootstrap: weighted APCD equals metrics.evaluate with weights 1, and a doubled frame equals the frame twice
    from clocs_r import bootstrap
    rng_b = np.random.default_rng(3)
    fr = []
    for _ in range(25):
        g = np.array([box(rng_b.uniform(5, 190), rng_b.uniform(-20, 20)) for _ in range(rng_b.integers(0, 5))], np.float32).reshape(-1, 7)
        dt = np.concatenate([g + rng_b.normal(0, 1.0, g.shape).astype(np.float32) * [1, 1, 0, 0, 0, 0, 0],
                             np.array([box(rng_b.uniform(5, 190), rng_b.uniform(-20, 20)) for _ in range(3)], np.float32)])
        fr.append((g, dt, rng_b.uniform(0, 1, len(dt)).astype(np.float32)))
    mt = bootstrap.frame_matches(fr)
    ref = metrics.evaluate(fr, bins=[(50, 100), (100, 200), (50, 200)])
    for b in ("50-100", "100-200", "50-200", "all"):
        assert abs(bootstrap.weighted_apcd(mt[b], np.ones(len(fr))) - ref[b]["APCD"]) < 1e-9, b
    w2 = np.ones(len(fr))
    w2[4] = 2
    w2[7] = 0
    dup = [f for i, f in enumerate(fr) if i != 7] + [fr[4]]
    ref2 = metrics.evaluate(dup, bins=[(50, 100), (100, 200), (50, 200)])
    assert abs(bootstrap.weighted_apcd(mt["50-200"], w2) - ref2["50-200"]["APCD"]) < 1e-9

    # Phase 6b radar filters: cues, the fast APCD, "last" merging below every base box, and persistence by ego motion
    from clocs_r import radar_filters as rflt
    c = dict(boxes=np.array([box(60, 0), box(120, 2), box(30, -3)], np.float32), scores=np.array([0.4, 0.08, 0.6], np.float32),
             keep_eval=np.array([True, True, False]), true=np.array([True, False, True]), moving=np.array([False, True, True]),
             lidar=np.array([True, False, False]), persist=np.array([False, True, False]))
    assert rflt.mask(c, 0.05, "none").tolist() == [True, True, False] and rflt.mask(c, 0.1, "none").tolist() == [True, False, False]
    assert rflt.mask(c, 0.05, "moving|persist").tolist() == [False, True, False] and rflt.mask(c, 0.05, "lidar").tolist() == [True, False, False]
    tab = rflt.precision_table([c])
    assert tab["t0.05|none"]["kept"] == [0, 1, 1] and tab["t0.05|none"]["true"] == [0, 1, 0]
    gt = np.array([box(60, 0.3), box(150, 0)], np.float32)
    bf = [(gt, np.array([box(150.5, 0), box(80, 5)], np.float32), np.array([0.9, 0.3], np.float32), "urban", "s")]
    frames6 = [(gt, bf[0][1], bf[0][2])]
    assert abs(rflt.apcd(frames6)["50-200"] - metrics.evaluate(frames6, bins=rflt.FAR_BINS)["50-200"]["APCD"]) < 1e-12
    last = rflt.merged(bf, [c], 0.05, "none", "last")
    assert len(last[0][1]) == 4 and last[0][2][:2].tolist() == [np.float32(0.9), np.float32(0.3)] and last[0][2][2:].max() < 0.3
    assert rflt.apcd([f[:3] for f in last])["50-200"] >= rflt.apcd(frames6)["50-200"]
    cal = rflt.merged(bf, [c], 0.05, "none", "calibrated", [0.0, 0.5, 0.2])
    assert sorted(cal[0][2].tolist()) == sorted([np.float32(0.9), np.float32(0.3), np.float32(0.5), np.float32(0.2)])

    class PoseSeq:
        name = "p"
        meta = dict(poses={7: np.eye(4), 6: np.array([[1, 0, 0, -3.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])})
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "p"))
        np.savez(os.path.join(d, "p", f"{6:07d}.npz"), boxes=np.array([box(63, 0), box(90, 0)], np.float32), scores=np.array([0.3, 0.01], np.float32))
        pm = rflt.persist(PoseSeq(), 7, d, np.array([box(60, 0.5), box(87, 0)], np.float32))
        assert pm.tolist() == [True, False]  # (63, 0) one frame ago is (60, 0) now; the 0.01 box is below RD_MIN
        assert rflt.persist(PoseSeq(), 6, d, np.array([box(60, 0)], np.float32)).tolist() == [False]  # no frame 5

    # Splits: recording names, and select/fit partition the training scenes with no recording on both sides
    from clocs_r import splits
    assert splits.recording("/x/train/night/20210723-222803-00.12.30-00.12.45@Yoda/") == "20210723-222803@Yoda"
    train = splits.scenes("train") + splits.scenes("holdout")
    if train:  # only where the dataset is present
        sel, fit = splits.scenes("select"), splits.scenes("fit")
        assert sorted(sel + fit) == sorted(train) and not set(sel) & set(fit)
        assert not {splits.recording(s) for s in sel} & {splits.recording(s) for s in fit}
        assert {s.split(os.sep)[-2] for s in sel} == set(splits.ODDS)
        folds = [splits.scenes(f"fold{k}") for k in range(splits.FOLDS)]
        assert sorted(sum(folds, [])) == sorted(fit)
        for k, f in enumerate(folds):
            rest = splits.scenes(f"notfold{k}")
            assert sorted(f + rest) == sorted(fit) and not {splits.recording(s) for s in f} & {splits.recording(s) for s in rest}
    print("all core checks passed")


if __name__ == "__main__":
    main()
