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
    --model mnist \
    --manifold PoincareBall \
    --c 1.0 \
    --K 1 \
    --latent-dim 2 \
    --hidden-dim 600 \
    --dec Geo \
    --enc Wrapped \
    --prior RiemannianNormal \
    --posterior RiemannianNormal \
    --epochs 80 \
    --save-freq 5 \
    --lr 5e-4 \
    --batch-size 128 \
    --iwae-samples 5000
