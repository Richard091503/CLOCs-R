# Source this file to set up the CLOCs environment:  source ~/Documents/CLOCs/env.sh
source ~/miniforge3/etc/profile.d/conda.sh
conda activate clocs
export CUDA_HOME=$HOME/cuda-11.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$HOME/cudnn-8.5/lib:$LD_LIBRARY_PATH
# numba needs to find the CUDA toolkit's libNVVM/libdevice
export NUMBAPRO_NVVM=$CUDA_HOME/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=$CUDA_HOME/nvvm/libdevice
export PYTHONPATH=$HOME/Documents/CLOCs:$PYTHONPATH
# numba 0.56 binds the wrong cuCtxGetDevice_v2 symbol on CUDA 13 drivers; use NVIDIA's cuda-python binding instead
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
