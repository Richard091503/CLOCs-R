#!/usr/bin/env bash
# Installs the two detector toolboxes named in the paper into the existing `clocs` environment:
#   OpenPCDet (SECOND, commit pinned below) and MMDetection 2.28 (Cascade R-CNN).
# Run once from any directory: bash ~/Documents/CLOCs/clocs_r/install_detectors.sh
# Activate first: conda's activation scripts read unset variables, which `set -u` would reject
source "$HOME/Documents/CLOCs/env.sh"
set -euo pipefail
OPENPCDET_COMMIT=233f849829b6ac19afb8af8837a0246890908755

# The conda GCC 9 toolchain is the one torch 1.10 and CUDA 11.3 were built against
export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++ CUDAHOSTCXX=x86_64-conda-linux-gnu-g++
export LDSHARED="x86_64-conda-linux-gnu-gcc -pthread -shared"

echo "== OpenPCDet"
[ -d "$HOME/OpenPCDet" ] || git clone https://github.com/open-mmlab/OpenPCDet.git "$HOME/OpenPCDet"
cd "$HOME/OpenPCDet"
git checkout -q "$OPENPCDET_COMMIT"
pip install -q easydict pyquaternion SharedArray pyyaml
# The Argoverse 2 module needs av2 and kornia at import time; CLOCs-R never uses it, so drop its registration
sed -i "/argo2/d; /Argo2/d" pcdet/datasets/__init__.py
# The spconv 1.2.1 built for this environment does not define __version__
sed -i "s/if float(spconv.__version__\[2:\]) >= 2.2:/if float(getattr(spconv, '__version__', '1.2')[2:]) >= 2.2:/" pcdet/utils/spconv_utils.py
python setup.py build_ext --inplace -j8 > /tmp/openpcdet_build.log 2>&1 || { tail -30 /tmp/openpcdet_build.log; exit 1; }
find pcdet/ops -name "*.so"

echo "== MMDetection"
pip install -q "yapf==0.40.1"  # newer yapf breaks mmcv 1.x config printing
pip install -q "mmcv-full==1.7.2" -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install -q "mmdet==2.28.2"

echo "== Checks"
cd "$HOME"
PYTHONPATH="$HOME/OpenPCDet:$PYTHONPATH" python -W ignore -c "
import torch, pcdet, mmcv, mmdet
from pcdet.models import build_network
from mmcv.ops import nms
print('pcdet', pcdet.__version__, '| mmcv', mmcv.__version__, '| mmdet', mmdet.__version__)
print('GPU nms ok:', nms(torch.tensor([[0,0,10,10],[1,1,11,11.]]).cuda(), torch.tensor([.9,.8]).cuda(), .5)[1].tolist())
"
echo "INSTALL OK"
