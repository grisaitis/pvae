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
    --model tree \
    --manifold PoincareBall \
    --latent-dim 2 \
    --hidden-dim 200 \
    --prior-std 1.7 \
    --c 1.2 \
    --data-size 50 \
    --data-params 6 2 1 1 5 5 \
    --dec Geo \
    --enc Wrapped \
    --prior RiemannianNormal \
    --posterior RiemannianNormal \
    --epochs 1000 \
    --save-freq 1000 \
    --lr 1e-3 \
    --batch-size 64 \
    --iwae-samples 5000
