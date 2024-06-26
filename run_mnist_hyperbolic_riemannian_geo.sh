#!/bin/bash

docker \
    run \
    --gpus all \
    -d \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    -u $(id -u):$(id -g) \
    pvae \
    python -u pvae/main.py \
    --model mnist \
    --manifold PoincareBall \
    --save-freq 0 \
    --epochs 80 \
    --batch-size 128 \
    --lr 5e-4 \
    --latent-dim 2 \
    --c 1.4 \
    --posterior RiemannianNormal \
    --hidden-dim 600 \
    --enc Wrapped \
    --dec Geo \
    --prior RiemannianNormal \
    --iwae-samples 5000 \
    --seed 42
