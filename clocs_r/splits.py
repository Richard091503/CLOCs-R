"""Scene splits and the label classes used throughout CLOCs-R."""
import os

from clocs_r.aimotive import DEFAULT_ROOT, Sequence, list_sequences

CAR_CLASSES = ("CAR", "VAN", "PICKUP")
# Large vehicles are neither targets nor false positives, as KITTI treats classes similar to Car
IGNORE_CLASSES = ("TRUCK", "BUS", "TRAILER")
# Driving conditions (the aiMotive folder names). Daytime clear weather is highway and urban; night and rain are the
# adverse conditions. Results before 2026-11-14 used CLEAR_ODDS only; CLOCS_ODDS=highway,urban reproduces that setting.
CLEAR_ODDS = ("highway", "urban")
ADVERSE_ODDS = ("night", "rain")
ODDS = tuple(os.environ.get("CLOCS_ODDS", ",".join(CLEAR_ODDS + ADVERSE_ODDS)).split(","))
# Training scenes held out for hyperparameter and feature-set selection (Section II-C/E): every 10th per driving condition,
# so each condition, rain included (4 training scenes), keeps at least one holdout scene
HOLDOUT_EVERY = 10
# Scenes with highway tunnels, where the camera is badly underexposed (mean luminance below 10/255 in sampled frames).
# None of the training scenes contain one, so results are also reported without these.
TUNNEL_SCENES = ("20210401-074958-00.01.45-00.02.00@Jarvis", "20210401-074958-00.01.30-00.01.45@Jarvis",
                 "20210401-074452-00.01.45-00.02.00@Jarvis", "20210401-085844-00.01.00-00.01.15@Jarvis",
                 "20210401-074452-00.01.30-00.01.45@Jarvis", "20210401-073402-00.18.00-00.18.15@Jarvis")


# Selection split (docs/MODELS.md, Splits): whole training recordings, so no drive is on both sides,
# chosen per condition for about 15% of its training frames and 20% of its cars at 100-200 m. fit is every other
# aiMotive training scene, the old holdout scenes included. train, holdout and val above are unchanged.
SELECT_RECORDINGS = {
    "highway": ("20210401-120439@Jarvis", "20220202-105828@Cali", "20220204-100251@Cali"),
    "urban": ("20210625-175142@Yoda", "20210630-130826@Yoda", "20210724-064134@Yoda", "20210729-162718@Yoda"),
    "night": ("20210701-225716@Yoda", "20210729-223722@Yoda", "20210817-214850@Yoda"),
    "rain": ("20210427-130007@Jarvis",),
}


def recording(scene):
    """Recording of a scene path or name: '20210723-222803-00.12.30-00.12.45@Yoda' -> '20210723-222803@Yoda'."""
    name = os.path.basename(scene.rstrip(os.sep))
    stem, vehicle = name.split("@")
    return "-".join(stem.split("-")[:2]) + "@" + vehicle


# Cross-fitting folds of fit (used on the thesis branch only): whole recordings, assigned per condition from
# the largest down, each to the fold with the fewest frames of that condition so far (ties to the lower index).
# foldK is fold K; notfoldK is the rest of fit, on which the detectors that score foldK are trained.
FOLDS = 3


def _fold_of(root):
    """Recording -> fold index, for every recording of fit."""
    out = {}
    for odd in ODDS:
        frames = {}
        for s in scenes("fit", root):
            if s.rstrip(os.sep).split(os.sep)[-2] == odd:
                r = recording(s)
                frames[r] = frames.get(r, 0) + len(Sequence(s).kept)
        load = [0] * FOLDS
        for r, n in sorted(frames.items(), key=lambda kv: (-kv[1], kv[0])):
            k = min(range(FOLDS), key=lambda i: (load[i], i))
            load[k] += n
            out[r] = k
    return out


def scenes(split, root=DEFAULT_ROOT):
    """split: train (detectors and fusion), holdout (selection) or val (reported results);
    select and fit: the next-round selection split and the training scenes outside it;
    foldK and notfoldK (K < FOLDS): a cross-fitting fold of fit and the rest of fit."""
    if split == "val":
        return list_sequences(root, "val", ODDS)
    if split in ("select", "fit"):
        return [s for odd in ODDS for s in list_sequences(root, "train", (odd,))
                if (recording(s) in SELECT_RECORDINGS.get(odd, ())) == (split == "select")]
    if split.startswith(("fold", "notfold")):
        k = int(split[len("notfold"):] if split.startswith("notfold") else split[len("fold"):])
        assert 0 <= k < FOLDS, split
        fold = _fold_of(root)
        return [s for s in scenes("fit", root) if (fold[recording(s)] == k) == split.startswith("fold")]
    assert split in ("train", "holdout"), split
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
