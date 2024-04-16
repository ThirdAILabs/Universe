# Ubuntu 20.04 (focal)
# https://hub.docker.com/_/ubuntu/?tab=tags&name=focal
ARG ROOT_CONTAINER=ubuntu:focal
FROM $ROOT_CONTAINER

# This line requires that we build the docker from the top-level directory of Universe
ARG KAGGLE_CREDS=docker/demos/kaggle.json

LABEL Description="Build Environment"
ENV HOME /root
SHELL ["/bin/bash", "-c"]

RUN apt update && DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install \
    git \
    curl \
    wget \
    libssl-dev \
    libcurl4-openssl-dev \
    # Install python and pip
    python3 \
    python3-dev \
    pip \
    # Install sudo
    && apt-get update && apt-get -y install sudo \
    #  The following is needed for building some pip packages (notably pytrec-eval, which is needed by beir)
    && sudo apt-get -y install build-essential

COPY $KAGGLE_CREDS $HOME/.kaggle/kaggle.json
ADD bin/run_single_demo_notebook.py run_single_demo_notebook.py  
RUN pip3 install jupyter==1.0.0 nbconvert==7.2.1 nbformat==5.7.0
