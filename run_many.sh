#!/bin/bash

for i in {1..5}; do
    ./run_rnaseq_hyperbolic_riemannian_geo_c0.25.sh
    sleep 10
    while [ $(docker ps -q | wc -l) -ge 4 ]; do
        sleep 1
    done
    ./run_rnaseq_hyperbolic_riemannian_geo_c1.4.sh
    sleep 10
    while [ $(docker ps -q | wc -l) -ge 4 ]; do
        sleep 1
    done
done
