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
    --manifold PoincareBall \
    --c 0.25 \
    --latent-dim 2 \
    --hidden-dim 50 \
    --prior-std 1.0 \
    --data-size 17324 \
    --dec Geo \
    --enc Wrapped \
    --prior WrappedNormal \
    --posterior WrappedNormal \
    --epochs 512 \
    --save-freq 512 \
    --lr 5e-4 \
    --batch-size 128 \
    --iwae-samples 5000
