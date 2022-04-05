# Ubuntu 20.04 (focal)
# https://hub.docker.com/_/ubuntu/?tab=tags&name=focal
ARG ROOT_CONTAINER=ubuntu:focal
FROM $ROOT_CONTAINER

LABEL Description="Build Environment"
ENV HOME /root
SHELL ["/bin/bash", "-c"]
RUN apt update ; DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install \
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
    ; apt-get update ; apt-get -y install sudo ; \
    pip3 install \
    dark \
    pytest \
    mlflow \
    toml \
    psutil \
    ipython \
    datasets \
    mmh3 \
    ; echo "export PYTHONPATH=/Universe/build:\$PYTHONPATH" >> $HOME/.bashrc \
    ; ln -s /usr/bin/clang-tidy-12 /usr/bin/clang-tidy \
    ; echo "export LSAN_OPTIONS=suppressions=/Universe/leak-suppressions.txt" >> $HOME/.bashrc