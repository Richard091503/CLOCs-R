#!/usr/bin/env bash
# Builds the `clocs` conda environment and the two detector toolboxes named in the paper:
#   OpenPCDet (SECOND, commit pinned below) and MMDetection 2.28 (Cascade R-CNN).
# Tested on an RTX 4090 (sm_89, Ubuntu 22.04) and a Jetstream A100 vGPU (sm_80, Ubuntu 24.04). Run once: bash clocs_r/install_detectors.sh
# The GPU architecture is read from nvidia-smi (override with TORCH_CUDA_ARCH_LIST), and GCC 11 comes from conda-forge,
# because CUDA 11.8 nvcc rejects host compilers newer than GCC 11 (Ubuntu 24.04 ships GCC 13).
# Needs miniforge in ~/miniforge3 (https://github.com/conda-forge/miniforge).
set -eo pipefail
OPENPCDET_COMMIT=233f849829b6ac19afb8af8837a0246890908755
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda env list | grep -q "^clocs " || conda create -y -q -n clocs python=3.10
conda activate clocs
# Packages in ~/.local (pip --user) and ROS's PYTHONPATH would otherwise shadow the environment's own
conda env config vars set PYTHONNOUSERSITE=1 > /dev/null
GPU_ARCH="${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)}"
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="$GPU_ARCH" MAX_JOBS="$(nproc)"
echo "== Building CUDA ops for compute capability $TORCH_CUDA_ARCH_LIST"

echo "== PyTorch 2.0.1 + CUDA 11.8, and nvcc 11.8 for the OpenPCDet ops"
pip install -q torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
conda install -y -q -c "nvidia/label/cuda-11.8.0" cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-cccl > /dev/null
conda install -y -q -c conda-forge "gcc=11" "gxx=11" > /dev/null  # nvcc 11.8 supports host GCC <= 11
gcc --version | head -1
pip install -q "numpy==1.24.4" spconv-cu118==2.3.6 "yapf==0.40.1"  # newer yapf breaks mmcv 1.x config printing

echo "== MMDetection"
pip install -q mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
pip install -q mmdet==2.28.2 "numba<0.59" scipy shapely "laspy[lazrs]" easydict pyquaternion SharedArray pyyaml tqdm \
    tensorboardX matplotlib requests fire scikit-image "opencv-python<4.9" "opencv-python-headless<4.9" "setuptools<70" "numpy==1.24.4"

echo "== OpenPCDet"
[ -d "$HOME/OpenPCDet" ] || git clone https://github.com/open-mmlab/OpenPCDet.git "$HOME/OpenPCDet"
cd "$HOME/OpenPCDet"
git checkout -q "$OPENPCDET_COMMIT"
# The Argoverse 2 module needs av2 and kornia at import time; CLOCs-R never uses it, so drop its registration
sed -i "/argo2/d; /Argo2/d" pcdet/datasets/__init__.py
# No -j: parallel extension builds share one ninja file and race; ninja already builds each extension in parallel
python setup.py build_ext --inplace > /tmp/openpcdet_build.log 2>&1 || { tail -30 /tmp/openpcdet_build.log; exit 1; }
find pcdet/ops -name "*.so"

echo "== Checks"
source "$REPO/env.sh"
cd "$HOME"
python -W ignore -c "
import torch, pcdet, mmcv, mmdet, spconv
from pcdet.models import build_network
from mmcv.ops import nms
print('torch', torch.__version__, '| spconv', spconv.__version__, '| pcdet', pcdet.__version__, '| mmcv', mmcv.__version__, '| mmdet', mmdet.__version__)
print('GPU nms ok:', nms(torch.tensor([[0,0,10,10],[1,1,11,11.]]).cuda(), torch.tensor([.9,.8]).cuda(), .5)[1].tolist())
"
echo "INSTALL OK"
