# Don't use this Dockerfile for development, changes won't be propogated
# back to your local copy of Universe

# Ubuntu 20.04 (focal)
# https://hub.docker.com/_/ubuntu/?tab=tags&name=focal
ARG ROOT_CONTAINER=ubuntu:focal
FROM $ROOT_CONTAINER

LABEL Description="Build Environment"
ENV HOME /root
SHELL ["/bin/bash", "-c"]

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
    clang-tidy-12 \
    clang \
    cmake \
    clang-format \
    gdb \
    htop \
    tmux \
    # Install python and pip
    python3 \
    python3-dev \
    pip \
    # Install sudo
    && apt-get update && apt-get -y install sudo

RUN python3 -m pip install --upgrade pip 

# This means the Dockerfile needs to be built from the root directory
# This will also copy everything, including files not tracked by git (build
# files, venv, etc). 
COPY . /Universe/

RUN rm -rf /Universe/build
RUN /Universe/bin/build.py --extras benchmark

# The exit 0 means these commands will still continue even if the install fails. 
# This is because these downloads might not work on arm64, but we still want
# to be able to build a docker image with some maximum amount of functionality
RUN sudo apt install openssh-client ; exit 0
RUN apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r` ; exit 0


RUN ln -s /usr/bin/clang-tidy-12 /usr/bin/clang-tidy \
    ; echo "export LSAN_OPTIONS=suppressions=/Universe/leak-suppressions.txt" >> $HOME/.bashrc
