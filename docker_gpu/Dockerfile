FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

RUN pip install -U pip && \
    pip install scikit-learn==0.21.3 scipy==1.3.2 seaborn==0.9.0

RUN pip install git+https://github.com/geoopt/geoopt.git@010f5bdf1d6cf217fb47a767e2d93e39369e4cb2#egg=geoopt

ENV MPLCONFIGDIR=/tmp/matplotlib
