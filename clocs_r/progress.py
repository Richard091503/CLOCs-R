"""Show the state of every CLOCs-R job and redraw a loss plot.

    python -m clocs_r.progress               # one snapshot
    watch -n 30 python -m clocs_r.progress   # live view; Ctrl-C stops watching, not the jobs

The plot is written to runs/progress.png; open it in VS Code and it refreshes as the file changes.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
RUNS = "runs"


def running(pattern):
    return subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL).returncode == 0


def duration(sec):
    return f"{int(sec // 3600)}h {int(sec % 3600 // 60):02d}m"


def second():
    path = f"{RUNS}/second/train.log"
    if not os.path.exists(path):
        return "not started", [], []
    steps, losses, last = [], [], None
    text = open(path).read()
    for m in re.finditer(r"epoch (\d+)/(\d+) iter (\d+)/(\d+) loss ([\d.]+).* ([\d.]+) it/s", text):
        last = m
        steps.append(int(m.group(3)))
        losses.append(float(m.group(5)))
    if last is None:
        return "starting", [], []
    it, total, rate = int(last.group(3)), int(last.group(4)), float(last.group(6))
    if "training finished" in text:
        return f"DONE, final loss {losses[-1]:.3f}", steps, losses
    state = (f"epoch {last.group(1)}/{last.group(2)}  {100 * it / total:5.1f}%  loss {losses[-1]:.3f}  "
             f"{rate:.1f} it/s  about {duration((total - it) / max(rate, 1e-3))} left")
    return state + ("" if running("clocs_r[.]train_second") else "  ** NOT RUNNING **"), steps, losses


def cascade():
    path = f"{RUNS}/cascade/console.log"
    if not os.path.exists(path):
        return "not started", [], []
    rows = re.findall(r"Epoch \[(\d+)\]\[(\d+)/(\d+)\].*?eta: ([^,]+),.*? loss: ([\d.]+)", open(path).read())
    if not rows:
        return "starting", [], []
    per_epoch = int(rows[-1][2])
    steps = [(int(e) - 1) * per_epoch + int(i) for e, i, _, _, _ in rows]
    losses = [float(r[4]) for r in rows]
    e, i, n, eta, loss = rows[-1]
    if not running("clocs_r[.]cascade_2d"):
        finished = os.path.exists(f"{RUNS}/cascade/epoch_4.pth")
        return (f"DONE, final loss {loss}" if finished else f"epoch {e}  ** NOT RUNNING **"), steps, losses
    return f"epoch {e}/4  iter {i}/{n}  loss {loss}  about {eta} left", steps, losses


def counted(root, ext):
    return len(glob.glob(f"{root}/*/*.{ext}"))


def main():
    sec, s_steps, s_loss = second()
    cas, c_steps, c_loss = cascade()
    print(time.strftime("%a %H:%M:%S"))
    print(f"SECOND (3D)            {sec}")
    print(f"Cascade R-CNN (2D)     {cas}")
    print(f"3D candidates cached   {counted(RUNS + '/candidates3d', 'npz')} frames")
    print(f"2D candidates cached   {counted(RUNS + '/candidates2d', 'npy')} frames")
    done = sorted(glob.glob(f"{RUNS}/fusion/*/results.json"))
    print(f"Fusion runs finished   {len(done)}")
    for path in done:
        r = json.load(open(path))
        if "val" in r:
            v = r["val"]["overall"]
            print(f"  {os.path.basename(os.path.dirname(path)):16s} val center-distance AP: "
                  f"100-200 m {v['100-200']['APCD']:.3f}   50-100 m {v['50-100']['APCD']:.3f}   all {v['all']['APCD']:.3f}")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print(f"GPU                    {gpu}")
    print(f"Disk free              {shutil.disk_usage('/').free / 1e9:.1f} GB")

    if s_steps or c_steps:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
        for ax, x, y, color, title in ((axes[0], s_steps, s_loss, BLUE, "SECOND training loss"),
                                       (axes[1], c_steps, c_loss, ORANGE, "Cascade R-CNN training loss")):
            ax.set_facecolor(SURFACE)
            ax.plot(x, y, color=color, linewidth=2)
            ax.set_title(title, color=INK, fontsize=11, loc="left")
            ax.set_xlabel("Training iteration", color=INK_2)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.tick_params(colors=INK_2, labelsize=9)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{RUNS}/progress.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
