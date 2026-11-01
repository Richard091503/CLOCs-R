"""Readers for the reduced aiMotive dataset written by fetch_aimotive.py."""
import collections
import glob
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_ROOT = os.path.expanduser("~/Documents/aimotive_clear")
# Detection region shared with the aiMotive baselines: forward half of their [-204.8, 204.8] x [-25.6, 25.6] m grid
REGION = dict(x=(0.0, 204.8), y=(-25.6, 25.6))
DIST_BINS = [(0, 30), (30, 50), (50, 100), (100, 200)]


class Sequence:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            self.meta = pickle.load(f)
        self.name, self.split, self.odd = self.meta["seq"], self.meta["split"], self.meta["odd"]
        self.kept = self.meta["kept"]
        self.cam_from_body = self.meta["cam_from_body"]
        self.body_from_radar = np.linalg.inv(self.meta["radar_from_body"])
        self._radar = None
        self._period = None

    @property
    def radar(self):
        if self._radar is None:
            with np.load(os.path.join(self.path, "radar.npz")) as z:
                self._radar = {k: z[k] for k in z.files}
                self._radar["index"] = {int(f): i for i, f in enumerate(self._radar["frames"])}
        return self._radar

    def lidar(self, frame):
        with np.load(os.path.join(self.path, "lidar", f"{frame:07d}.npz")) as z:
            return np.hstack([z["xyz"] / 100.0, z["intensity"][:, None] / 255.0]).astype(np.float32)

    def image_path(self, frame):
        return os.path.join(self.path, "image", f"{frame:07d}.jpg")

    def boxes(self, frame, classes=None):
        """Labelled boxes in the body frame: (N, 7) [x, y, z_center, l, w, h, yaw], plus names and relative velocity."""
        objs = self.meta["labels"].get(frame, [])
        if classes is not None:
            objs = [o for o in objs if o["ObjectType"] in classes]
        boxes = np.zeros((len(objs), 7), dtype=np.float32)
        for i, o in enumerate(objs):
            yaw = Rotation.from_quat([o[f"BoundingBox3D Orientation Quat {a}"] for a in "XYZW"]).as_euler("xyz")[2]
            boxes[i] = [o["BoundingBox3D Origin X"], o["BoundingBox3D Origin Y"], o["BoundingBox3D Origin Z"],
                        o["BoundingBox3D Extent X"], o["BoundingBox3D Extent Y"], o["BoundingBox3D Extent Z"], yaw]
        return boxes, [o["ObjectType"] for o in objs]

    def project(self, pts):
        """Body-frame points (..., 3) -> pixel coordinates (..., 2) and camera depth, using the opencv_pinhole model."""
        R, t = self.cam_from_body[:3, :3], self.cam_from_body[:3, 3]
        c = pts @ R.T + t
        z = c[..., 2]
        zs = np.where(np.abs(z) < 1e-3, 1e-3, z)
        x, y = c[..., 0] / zs, c[..., 1] / zs
        k1, k2, p1, p2, k3 = self.meta["dist"]
        r2 = x * x + y * y
        d = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * d + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * d + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        fx, fy = self.meta["focal"]
        cx, cy = self.meta["principal"]
        return np.stack([fx * xd + cx, fy * yd + cy], -1), z

    def ego_velocity(self, frame):
        """Ego velocity in the body frame at this frame (m/s), by central difference of the GNSS-INS poses."""
        poses, times = self.meta["poses"], self.meta["times"]
        for a, b in ((frame - 1, frame + 1), (frame, frame + 1), (frame - 1, frame)):
            if a in poses and b in poses:
                # A few frames have a pose but no timestamp; the sequence's frame period stands in for those
                dt = times[b] - times[a] if a in times and b in times and times[b] > times[a] else (b - a) * self.frame_period
                return (np.linalg.inv(poses[a]) @ poses[b])[:3, 3] / dt
        return np.zeros(3)

    @property
    def frame_period(self):
        if self._period is None:
            f = np.array(sorted(self.meta["times"]))
            t = np.array([self.meta["times"][i] for i in f])
            self._period = float(np.median(np.diff(t) / np.diff(f)))
        return self._period

    def radar_sweep(self, frame):
        """One sweep in the body frame: (N, 3) xyz and ego-compensated radial velocity (Eq. 1)."""
        rad = self.radar
        if frame not in rad["index"]:
            return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
        i = rad["index"][frame]
        a, b = rad["offsets"][i], rad["offsets"][i + 1]
        r, az, el = rad["range"][a:b], rad["azimuth"][a:b], rad["elevation"][a:b]
        s = np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)], -1)
        T = self.body_from_radar
        p = s @ T[:3, :3].T + T[:3, 3]
        u = self.ego_velocity(frame)[:2]
        n = p[:, :2] / np.maximum(np.linalg.norm(p[:, :2], axis=1, keepdims=True), 1e-3)
        return p.astype(np.float32), (rad["speed"][a:b] + n @ u).astype(np.float32)

    def radar_accumulated(self, frame, m=3):
        """Returns from the last m sweeps, moved into this frame's body coordinates using the ego poses."""
        poses = self.meta["poses"]
        pts, vel = [], []
        for f in range(frame, frame - m, -1):
            p, v = self.radar_sweep(f)
            if len(p) and f != frame:
                if f not in poses or frame not in poses:
                    continue
                T = np.linalg.inv(poses[frame]) @ poses[f]
                p = p @ T[:3, :3].T + T[:3, 3]
            pts.append(p)
            vel.append(v)
        return np.concatenate(pts).astype(np.float32), np.concatenate(vel).astype(np.float32)


def list_sequences(root=DEFAULT_ROOT, split=None, odds=None):
    out = []
    for meta in sorted(glob.glob(os.path.join(root, "*", "*", "*", "meta.pkl"))):
        seq = os.path.dirname(meta)
        s, o = seq.split(os.sep)[-3:-1]
        if (split is None or s == split) and (odds is None or o in odds):
            out.append(seq)
    return out


def box_corners_bev(boxes):
    """(N, 7) boxes -> (N, 4, 2) BEV corners."""
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    dx, dy = boxes[:, 3] / 2, boxes[:, 4] / 2
    signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float32)  # counter-clockwise
    x = signs[None, :, 0] * dx[:, None]
    y = signs[None, :, 1] * dy[:, None]
    return np.stack([boxes[:, None, 0] + x * c[:, None] - y * s[:, None],
                     boxes[:, None, 1] + x * s[:, None] + y * c[:, None]], -1)


def box_corners_3d(boxes):
    """(N, 7) boxes -> (N, 8, 3) corners."""
    bev = box_corners_bev(boxes)
    z0 = boxes[:, None, 2] - boxes[:, None, 5] / 2
    z1 = boxes[:, None, 2] + boxes[:, None, 5] / 2
    return np.concatenate([np.concatenate([bev, z0[..., None] * np.ones((len(boxes), 4, 1))], -1),
                           np.concatenate([bev, z1[..., None] * np.ones((len(boxes), 4, 1))], -1)], 1)


def project_boxes(seq, boxes):
    """Tightest image rectangle of each 3D box, clipped to the image: (N, 4) pixels and a validity mask.

    Boxes with a corner behind the camera or a projection fully outside the image are invalid.
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, bool)
    uv, z = seq.project(box_corners_3d(boxes.astype(np.float64)))
    w, h = seq.meta["image_size"]
    rect = np.stack([uv[..., 0].min(1), uv[..., 1].min(1), uv[..., 0].max(1), uv[..., 1].max(1)], 1)
    front = (z > 0.1).all(1)
    clipped = rect.copy()
    clipped[:, [0, 2]] = np.clip(rect[:, [0, 2]], 0, w)
    clipped[:, [1, 3]] = np.clip(rect[:, [1, 3]], 0, h)
    visible = (clipped[:, 2] - clipped[:, 0] >= 2) & (clipped[:, 3] - clipped[:, 1] >= 2)
    return clipped.astype(np.float32), front & visible


def in_camera_view(seq, boxes, margin=0):
    """Mask of boxes whose center projects inside the image and in front of the camera."""
    if len(boxes) == 0:
        return np.zeros(0, bool)
    uv, z = seq.project(boxes[:, :3].astype(np.float64))
    w, h = seq.meta["image_size"]
    return (z > 0.5) & (uv[:, 0] >= -margin) & (uv[:, 0] < w + margin) & (uv[:, 1] >= -margin) & (uv[:, 1] < h + margin)


def in_region(boxes):
    return (boxes[:, 0] >= REGION["x"][0]) & (boxes[:, 0] < REGION["x"][1]) & \
           (boxes[:, 1] >= REGION["y"][0]) & (boxes[:, 1] < REGION["y"][1])
