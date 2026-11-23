# Source this file to set up the CLOCs-R environment:  source env.sh   (from the repository root, or by full path)
#
# Environment (RTX 4090 workstation; also Jetstream A100 vGPU): conda env `clocs` with Python 3.10, PyTorch 2.0.1 + CUDA 11.8, spconv-cu118 2.3.6,
# mmcv-full 1.7.2 / mmdet 2.28.2, and OpenPCDet built from source with the env's CUDA 11.8 nvcc (see clocs_r/install_detectors.sh).
CLOCS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate clocs
# Keep packages from ~/.local and ROS (/opt/ros/.../site-packages) out of this environment
export PYTHONNOUSERSITE=1
export PYTHONPATH="$CLOCS_ROOT:$HOME/OpenPCDet"
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)}"
# Reduced aiMotive dataset written by clocs_r/fetch_aimotive.py (all four driving conditions, every annotated frame)
export AIMOTIVE_ROOT="${AIMOTIVE_ROOT:-$HOME/datasets/aimotive_clocs}"
