#!/bin/bash

docker \
    run \
    --rm \
    --gpus all \
    -d \
    -v $(pwd):/workspaces/pvae \
    -w /workspaces/pvae \
    -u $(id -u):$(id -g) \
    pvae \
    python -u pvae/main.py \
    --model mnist \
    --manifold Euclidean \
    --save-freq 0 \
    --epochs 80 \
    --batch-size 128 \
    --lr 5e-4 \
    --latent-dim 2 \
    --posterior Normal \
    --hidden-dim 600 \
    --enc Wrapped \
    --dec Wrapped \
    --prior Normal \
    --iwae-samples 5000 \
    --seed 42