#!/bin/bash

docker \
    run \
    --gpus all \
    --rm \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    -u $(id -u):$(id -g) \
    pvae:1.3-cuda10.1-cudnn7-runtime \
    python -u pvae/main.py \
    --model rnaseq \
    --manifold Euclidean \
    --latent-dim 2 \
    --hidden-dim 200 \
    --prior-std 1.0 \
    --data-size 23686 \
    --dec Linear \
    --enc Linear \
    --prior Normal \
    --posterior Normal \
    --epochs 512 \
    --save-freq 512 \
    --lr 1e-3 \
    --batch-size 64 \
    --iwae-samples 5000
