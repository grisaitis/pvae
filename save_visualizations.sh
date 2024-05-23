#!/bin/bash

docker \
    run \
    --rm \
    --gpus all \
    -v $(pwd):/workspaces/pvae \
    -w /workspaces/pvae \
    -u $(id -u):$(id -g) \
    pvae:1.3-cuda10.1-cudnn7-runtime \
    python -u pvae/save_visualizations.py \
    --run-path experiments/2024-05-15T21_22_28.498641zoupnfno
