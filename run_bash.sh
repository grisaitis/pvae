#!/bin/bash

docker \
    run \
    --rm \
    --gpus all \
    -it \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    pvae:1.3-cuda10.1-cudnn7-runtime \
    bash