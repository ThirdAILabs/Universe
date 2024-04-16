# ARG ROOT_CONTAINER=ubuntu:focal
# FROM $ROOT_CONTAINER
FROM python:3.10-slim-bullseye

LABEL Description="Build Environment"
ENV HOME /root
SHELL ["/bin/bash", "-c"]

VOLUME /volume

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1
RUN pip install --upgrade pip

RUN apt update && DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install \
    # Install tools neccesary for building, writing, and running code
    build-essential \
    git \
    awscli\
    curl \
    wget \
    mpich \
    nano \
    vim \
    libmpich-dev \
    clang \
    cmake \
    clang-format \
    gdb \
    htop \
    tmux \
    libssl-dev \
    zlib1g-dev \
    libcurl4-openssl-dev \
    # Install sudo
    && apt-get update && apt-get -y install sudo

RUN python3 -m pip install --upgrade pip
RUN pip install cmake --upgrade

# This means the Dockerfile needs to be built from the root directory
# This will also copy everything, including files not tracked by git (build
# files, venv, etc). 
COPY . /Universe/

RUN rm -rf /Universe/build
RUN cd /Universe/ ; python3 setup.py bdist_wheel

CMD ["python3", "/Universe/docker/linuxarm64/move_to_volume.py", "/Universe/dist", "/volume"]