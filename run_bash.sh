#!/bin/bash

docker \
    run \
    --rm \
    --gpus all \
    -it \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    pvae \
    bash