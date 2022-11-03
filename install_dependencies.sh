#!/bin/bash

source $(conda info --root)/etc/profile.d/conda.sh

conda create -y --name gsk python=3.7.3
conda activate gsk
conda env update --file $PWD/environment.yml

TORCH=1.12.0
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    CUDA=cu113
    OS_VERS="linux-x86_64"
else
    CUDA=cpu
    OS_VERS="macosx_10_14_x86_64"
fi

pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"
#pip install "https://data.pyg.org/whl/torch-1.8.0%2B${CUDA}/torch_scatter-2.0.8-cp37-cp37m-${OS_VERS}.whl"
#pip install "https://data.pyg.org/whl/torch-1.8.0%2B${CUDA}/torch_sparse-0.6.12-cp37-cp37m-${OS_VERS}.whl"
#pip install "https://data.pyg.org/whl/torch-1.8.0%2B${CUDA}/torch_spline_conv-1.2.1-cp37-cp37m-${OS_VERS}.whl"
#pip install "https://data.pyg.org/whl/torch-1.8.0%2B${CUDA}/torch_cluster-1.5.9-cp37-cp37m-${OS_VERS}.whl"
pip install torch-geometric