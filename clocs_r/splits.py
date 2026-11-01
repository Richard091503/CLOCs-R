"""Scene splits and the label classes used throughout CLOCs-R."""
import os

from clocs_r.aimotive import DEFAULT_ROOT, Sequence, list_sequences

CAR_CLASSES = ("CAR", "VAN", "PICKUP")
# Large vehicles are neither targets nor false positives, as KITTI treats classes similar to Car
IGNORE_CLASSES = ("TRUCK", "BUS", "TRAILER")
# Daytime clear weather only: the aiMotive highway and urban folders (night and rain are left out)
ODDS = ("highway", "urban")
# Training scenes held out for hyperparameter and feature-set selection (Section II-C/E): every 10th per driving condition
HOLDOUT_EVERY = 10
# Scenes with highway tunnels, where the camera is badly underexposed (mean luminance below 10/255 in sampled frames).
# None of the training scenes contain one, so results are also reported without these.
TUNNEL_SCENES = ("20210401-074958-00.01.45-00.02.00@Jarvis", "20210401-074958-00.01.30-00.01.45@Jarvis",
                 "20210401-074452-00.01.45-00.02.00@Jarvis", "20210401-085844-00.01.00-00.01.15@Jarvis",
                 "20210401-074452-00.01.30-00.01.45@Jarvis", "20210401-073402-00.18.00-00.18.15@Jarvis")


def scenes(split, root=DEFAULT_ROOT):
    """split: train (detectors and fusion), holdout (selection) or val (reported results)."""
    if split == "val":
        return list_sequences(root, "val", ODDS)
    out = []
    for odd in ODDS:
        seqs = list_sequences(root, "train", (odd,))
        for i, s in enumerate(seqs):
            if (i % HOLDOUT_EVERY == 0) == (split == "holdout"):
                out.append(s)
    return out


def frames(split, root=DEFAULT_ROOT):
    """All (Sequence, frame) pairs of a split that have LiDAR and images on disk."""
    return [(seq, f) for seq in map(Sequence, scenes(split, root)) for f in seq.kept]
