#!/bin/bash

# iterate over all runs in experiments/ and call save_visualizations
for i in {1..5}; do
    while [ $(docker ps -q | wc -l) -ge 5 ]; do
        watch -t -n 0.1 'nvidia-smi && break'
    done
    ./run_synthetic_euclidean.sh
    ./run_synthetic_hyperbolic_wrapped_geo.sh
done