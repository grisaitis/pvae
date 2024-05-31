#!/bin/bash

function save_visualizations {
    docker \
        run \
        --rm \
        -d \
        --gpus all \
        -v $(pwd):$(pwd) \
        -w $(pwd) \
        -u $(id -u):$(id -g) \
        pvae:1.3-cuda10.1-cudnn7-runtime \
        python -u pvae/save_visualizations.py \
        --run-path $1
}

# iterate over all runs in experiments/ and call save_visualizations
for experiment in experiments/2024-05-30T20_*; do
    save_visualizations $experiment
    while [ $(docker ps -q | wc -l) -ge 5 ]; do
        echo "Waiting for containers to finish..."
        sleep 1
    done
done