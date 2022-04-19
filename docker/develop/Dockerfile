# Ubuntu 20.04 (focal)
# https://hub.docker.com/_/ubuntu/?tab=tags&name=focal
ARG ROOT_CONTAINER=ubuntu:focal
FROM $ROOT_CONTAINER

LABEL Description="Build Environment"
ENV HOME /root
SHELL ["/bin/bash", "-c"]

RUN apt update && DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install \
    build-essential \
    git \
    curl \
    mpich \
    libmpich-dev \
    clang-tidy-12 \
    clang \
    cmake \
    clang-format \
    gdb \
    python3 \
    python3-dev \
    pip \
    && apt-get update && apt-get -y install sudo
  
# Split this into a different RUN command for clarity, and also so if you want
# to add a python package you don't need to rebuild everything
RUN pip3 install \
    black \
    pytest \
    mlflow \
    toml \
    psutil \
    ipython \
    datasets \
    mmh3 \
    numpy \ 
    pandas

# The exit 0 means these commands will still continue even if the install fails, 
# this is because tensorflow (and possibly pytorch) won't work on arm64
RUN pip3 install tensorflow tensorflow-datasets ; exit 0
RUN pip3 install torch torchvision ; exit 0

RUN echo "export PYTHONPATH=/Universe/build:/Universe/benchmarks:\$PYTHONPATH" >> $HOME/.bashrc \
    ; ln -s /usr/bin/clang-tidy-12 /usr/bin/clang-tidy \
    ; echo "export LSAN_OPTIONS=suppressions=/Universe/leak-suppressions.txt" >> $HOME/.bashrc
