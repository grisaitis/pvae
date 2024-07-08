#!/bin/bash

docker \
    run \
    --gpus all \
    -d \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    -u $(id -u):$(id -g) \
    pvae:1.3-cuda10.1-cudnn7-runtime \
    python -u pvae/main.py \
    --model rnaseq \
    --manifold Euclidean \
    --K 5 \
    --latent-dim 2 \
    --hidden-dim 50 \
    --prior-std 1.0 \
    --data-size 17324 \
    --dec Linear \
    --enc Linear \
    --prior Normal \
    --posterior Normal \
    --epochs 256 \
    --save-freq 1 \
    --lr 5e-4 \
    --batch-size 128 \
    --iwae-samples 5000
