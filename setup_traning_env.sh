#!/usr/bin/env bash
conda install -c anaconda future
conda install pytorch torchvision -c pytorch
#conda install pytorch-cpu torchvision-cpu -c pytorch # Uncomment for linux without GPU
conda install -c conda-forge tensorflow
conda install -c conda-forge tensorboardx
