# Ubuntu 20.04 (focal)
# https://hub.docker.com/_/ubuntu/?tab=tags&name=focal
ARG ROOT_CONTAINER=ubuntu:focal
FROM $ROOT_CONTAINER

# This line requires that we build the docker from the top-level directory of Universe
ARG LICENSE=licensing/python_tests/license.serialized

ARG KAGGLE_CREDS=docker/demos/kaggle.json

LABEL Description="Build Environment"
ENV HOME /root
SHELL ["/bin/bash", "-c"]

RUN apt update && DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install \
    git \
    curl \
    wget \
    # Install python and pip
    python3 \
    python3-dev \
    pip \
    # Install sudo
    && apt-get update && apt-get -y install sudo

COPY $LICENSE $HOME/license.serialized
COPY $KAGGLE_CREDS $HOME/.kaggle/kaggle.json
ADD bin/run_demo_notebooks.py run_demo_notebooks.py  
RUN pip3 install thirdai
RUN pip3 install jupyter nbconvert nbformat
