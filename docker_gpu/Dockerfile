ARG BASE_IMAGE_TAG

FROM pytorch/pytorch:${BASE_IMAGE_TAG}

RUN apt update \
    && apt install -y git \
    && apt autoremove -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip && \
    pip install scikit-learn==0.21.3 scipy==1.3.2 seaborn==0.9.0
# pip install scikit-learn==1.3.2 scipy==1.10.1 seaborn==0.13.2

RUN pip install git+https://github.com/geoopt/geoopt.git@010f5bdf1d6cf217fb47a767e2d93e39369e4cb2#egg=geoopt

ENV MPLCONFIGDIR=/tmp/matplotlib
