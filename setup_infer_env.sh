#!/usr/bin/env bash
conda install -c conda-forge protobuf numpy
pip install onnx
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
conda install -c conda-forge scikit-image
