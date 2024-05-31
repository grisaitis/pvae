#!/bin/bash

docker \
    run \
    --rm \
    --gpus all \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    -u $(id -u):$(id -g) \
    pvae:1.3-cuda10.1-cudnn7-runtime \
    python -um pytest pvae/
