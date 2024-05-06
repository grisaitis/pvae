#!/bin/bash

docker \
    run \
    --rm \
    --gpus all \
    -it \
    -v $(pwd):/workspaces/pvae \
    -w /workspaces/pvae \
    pvae \
    bash