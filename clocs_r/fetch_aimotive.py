"""Fetch only the parts of the aiMotive dataset CLOCs-R needs, without downloading the archive.

The Kaggle mirror serves the 87 GB dataset as one zip on storage that honours HTTP range
requests, so we read the zip's index and pull individual files. For each clear-weather sequence
(daytime highway and urban; night and rain excluded) this keeps:

    meta.pkl        frame ids, kept frames, 3D labels, calibration and ego poses
    radar.npz       every front-radar target of every sweep in the sequence
    lidar/<f>.npz   LiDAR points inside the front camera's view (int16 cm xyz, uint8 intensity)
    image/<f>.jpg   front camera image

LiDAR and images are stored only for "kept" frames: every annotated val frame and every
`--train_stride`-th annotated train frame. Labels, radar and poses are kept for all frames.

Usage: python -m clocs_r.fetch_aimotive --out ~/Documents/aimotive_clear
"""
import argparse
import collections
import csv
import io
import json
import os
import pathlib
import pickle
import re
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import laspy
import numpy as np
import requests

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/tamasmatuszka/aimotive-multimodal-dataset"
CAMERA = "F_MIDLONGRANGECAM_CL"
RADAR = "F_LRR_C"
CLEAR_ODDS = ("highway", "urban")
MAX_TAN_FOV = 0.75  # |x/z| in camera coords; about 37 deg, just outside the undistorted image edge
MAX_RANGE = 210.0
MAX_GAP = 1 << 20  # merge wanted byte ranges closer than this into a single request


def signed_url():
    r = requests.get(KAGGLE_URL, stream=True, headers={"Range": "bytes=0-0"}, timeout=60)
    r.raise_for_status()
    r.close()
    return r.url


def read_index(url):
    """Parse the zip64 central directory -> {name: (local header offset, compressed size)}."""
    size = int(requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60).headers["Content-Range"].split("/")[1])
    tail = requests.get(url, headers={"Range": f"bytes={size - 65536 * 2}-{size - 1}"}, timeout=120).content
    i = tail.rfind(b"PK\x06\x06")  # zip64 end of central directory
    if i < 0:
        raise RuntimeError("zip64 end of central directory not found")
    cd_size, cd_off = struct.unpack("<QQ", tail[i + 40:i + 56])
    cd = requests.get(url, headers={"Range": f"bytes={cd_off}-{cd_off + cd_size - 1}"}, timeout=600).content
    index, p = {}, 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        comp, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nlen, xlen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        off = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nlen].decode()
        x = p + 46 + nlen
        extra = cd[x:x + xlen]
        q = 0
        while q + 4 <= len(extra):  # zip64 extended information holds whichever fields overflowed
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                fields, r = extra[q + 4:q + 4 + hlen], 0
                if usize == 0xFFFFFFFF:
                    r += 8
                if comp == 0xFFFFFFFF:
                    comp = struct.unpack("<Q", fields[r:r + 8])[0]
                    r += 8
                if off == 0xFFFFFFFF:
                    off = struct.unpack("<Q", fields[r:r + 8])[0]
            q += 4 + hlen
        index[name] = (off, comp)
        p = x + xlen + clen
    return index


class Archive:
    def __init__(self):
        self.url = signed_url()
        self.index = read_index(self.url)
        self.session = requests.Session()

    def get_many(self, names):
        """Return {name: decompressed bytes}, merging nearby ranges into single requests."""
        items = sorted(((self.index[n][0], self.index[n][1], n) for n in names))
        out, group = {}, []

        def flush():
            start = group[0][0]
            # Local headers carry a name plus an extra field of unknown length, hence the slack
            end = group[-1][0] + group[-1][1] + 30 + len(group[-1][2]) + 4096
            buf = self._range(start, end)
            for off, comp, name in group:
                h = off - start
                nlen, xlen = struct.unpack("<HH", buf[h + 26:h + 30])
                data = buf[h + 30 + nlen + xlen:h + 30 + nlen + xlen + comp]
                out[name] = zlib.decompress(data, -15)

        for it in items:
            if group and it[0] - (group[-1][0] + group[-1][1]) > MAX_GAP:
                flush()
                group = []
            group.append(it)
        if group:
            flush()
        return out

    def _range(self, start, end):
        for attempt in range(6):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
                if r.status_code in (401, 403):  # signed URL expired
                    self.url = signed_url()
                    continue
                r.raise_for_status()
                return r.content
            except (requests.RequestException, OSError):
                time.sleep(2 ** attempt)
        raise RuntimeError(f"failed to fetch bytes {start}-{end}")


def frame_id(name):
    return int(re.findall(r"(\d+)\.\w+$", name)[0])


def convert_sequence(arc, seq, names, out_root, train_stride):
    split, odd, seq_name = seq.split("/")
    out = pathlib.Path(out_root) / split / odd / seq_name
    if (out / "meta.pkl").exists():
        return 0
    box_names = sorted(n for n in names if "/dynamic/box/3d_body/" in n)
    annotated = [frame_id(n) for n in box_names]
    kept = annotated if split == "val" else annotated[::train_stride]

    p = f"{seq}/"
    small = list(box_names) + [p + "sensor/calibration/calibration.json", p + "sensor/camera/sync_frame2host.json",
                               p + "sensor/gnssins/egomotion2.json"]
    if p + "sensor/gnssins/egomotion2.json" not in arc.index:
        small[-1] = p + "sensor/gnssins/egomotion.json"
    # The archive holds a few stray hidden temp files in these folders (e.g. ".F_LRR_C_0012005.json.LBzr3o")
    radar_names = sorted(n for n in names if f"/sensor/radar/{RADAR}/" in n and n.endswith(".json") and "/." not in n)
    blobs = arc.get_many(small + radar_names)

    calib = json.loads(blobs[p + "sensor/calibration/calibration.json"])
    cam = calib[CAMERA]
    cam_from_body = np.array(cam["RT_sensor_from_body"], dtype=np.float64)
    radar_from_body = np.array(calib[RADAR]["RT_sensor_from_body"], dtype=np.float64)

    ego_raw = json.loads(blobs[small[-1]])
    radar_frames = [frame_id(n) for n in radar_names]
    span = range(min(radar_frames + annotated) - 2, max(radar_frames + annotated) + 3)
    host_ns = json.loads(blobs[p + "sensor/camera/sync_frame2host.json"])
    poses, times = {}, {}
    for f in span:
        v = ego_raw.get(str(f))
        if v is not None:  # egomotion2.json holds a dict, egomotion.json a flattened 4x4
            poses[f] = np.array(v["RT_ECEF_body"] if isinstance(v, dict) else v, dtype=np.float64).reshape(4, 4)
        if str(f) in host_ns:
            times[f] = host_ns[str(f)] * 1e-9

    labels = {}
    for n in box_names:
        labels[frame_id(n)] = json.loads(blobs[n])["CapturedObjects"]

    # Radar: one array per sweep in the radar's own frame, flattened with per-sweep offsets
    az, el, rng, speed, rcs, power, offs = [], [], [], [], [], [], [0]
    for n in radar_names:
        for t in json.loads(blobs[n])["targets"]:
            az.append(t["azimuth"]); el.append(t["elevation"]); rng.append(t["range"])
            speed.append(t["speed"]); rcs.append(t["rcs"]); power.append(t["power"])
        offs.append(len(az))
    (out / "lidar").mkdir(parents=True, exist_ok=True)
    (out / "image").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "radar.npz", frames=np.array(radar_frames, dtype=np.int64),
                        offsets=np.array(offs, dtype=np.int64),
                        **{k: np.array(v, dtype=np.float32) for k, v in
                           dict(azimuth=az, elevation=el, range=rng, speed=speed, rcs=rcs, power=power).items()})

    wanted = [p + f"dynamic/raw-revolutions/frame_{f:07d}.laz" for f in kept] + \
             [p + f"sensor/camera/{CAMERA}/{CAMERA}_{f:07d}.jpg" for f in kept]
    blobs = arc.get_many(wanted)
    R, t = cam_from_body[:3, :3], cam_from_body[:3, 3]
    for f in kept:
        with open(out / "image" / f"{f:07d}.jpg", "wb") as fp:
            fp.write(blobs[p + f"sensor/camera/{CAMERA}/{CAMERA}_{f:07d}.jpg"])
        las = laspy.read(io.BytesIO(blobs[p + f"dynamic/raw-revolutions/frame_{f:07d}.laz"]))
        xyz = np.vstack([las.x, las.y, las.z]).T  # already in the body frame
        c = xyz @ R.T + t
        keep = (c[:, 2] > 0.1) & (np.abs(c[:, 0]) < MAX_TAN_FOV * c[:, 2]) & (np.linalg.norm(xyz[:, :2], axis=1) < MAX_RANGE)
        np.savez_compressed(out / "lidar" / f"{f:07d}.npz",
                            xyz=np.round(xyz[keep] * 100).astype(np.int16),
                            intensity=np.clip(np.asarray(las.intensity)[keep], 0, 255).astype(np.uint8))
    meta = dict(split=split, odd=odd, seq=seq_name, annotated=annotated, kept=kept, labels=labels,
                poses=poses, times=times, cam_from_body=cam_from_body, radar_from_body=radar_from_body,
                focal=cam["focal_length_px"], principal=cam["principal_point_px"],
                image_size=cam["image_resolution_px"], dist=cam["distortion_coeffs"], cam_model=cam["model"])
    tmp = out / "meta.pkl.tmp"
    with open(tmp, "wb") as fp:
        pickle.dump(meta, fp)
    os.replace(tmp, out / "meta.pkl")  # last, so meta.pkl marks a finished sequence
    return len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="only this many sequences (for testing)")
    args = ap.parse_args()
    out = os.path.expanduser(args.out)

    arc = Archive()
    print("archive entries:", len(arc.index), flush=True)
    seqs = collections.defaultdict(list)
    for n in arc.index:
        parts = n.split("/")
        if len(parts) > 3 and parts[1] in CLEAR_ODDS:
            seqs["/".join(parts[:3])].append(n)
    todo = sorted(seqs)
    if args.limit:
        todo = todo[:: max(1, len(todo) // args.limit)][:args.limit]
    print("clear-weather sequences:", len(todo), flush=True)

    t0, done, frames = time.time(), 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(convert_sequence, arc, s, seqs[s], out, args.train_stride): s for s in todo}
        for fut in as_completed(futs):
            frames += fut.result()
            done += 1
            print(f"{done}/{len(todo)} {futs[fut]} frames={frames} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
