#!/bin/bash

# note: run this from the repo root

BASE_IMAGE_TAG=1.3-cuda10.1-cudnn7-runtime
# Oct 15, 2019 at 1:14 am, https://hub.docker.com/layers/pytorch/pytorch/1.3-cuda10.1-cudnn7-runtime/images/sha256-c3f6590d7263c67c8d1423badcd732607fa86c14e7acfa7843df25e07562828a?context=explore
# python 3.7, i think

# BASE_IMAGE_TAG=1.7.0-cuda11.0-cudnn8-runtime
# Oct 27, 2020 at 12:45 pm, https://hub.docker.com/layers/pytorch/pytorch/1.7.0-cuda11.0-cudnn8-runtime/images/sha256-9cffbe6c391a0dbfa2a305be24b9707f87595e832b444c2bde52f0ea183192f1?context=explore
# python 3.8

# BASE_IMAGE_TAG=2.0.0-cuda11.7-cudnn8-runtime
# Mar 20, 2023 at 2:22 pm
# python 3.10.9

docker \
    build \
    --build-arg BASE_IMAGE_TAG=${BASE_IMAGE_TAG} \
    -t pvae:${BASE_IMAGE_TAG} \
    $(pwd)/docker_gpu/
