#!/bin/bash

# note: run this from the repo root

docker build \
    -t pvae \
    $(pwd)/docker_gpu/
