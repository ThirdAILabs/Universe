FROM ubuntu:20.04
LABEL Description="Build Environment"
ENV HOME /root
SHELL ["/bin/bash", "-c"]
RUN apt update && DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install \
    build-essential \
    git \
    curl \
    clang \
    cmake \
    clang-tidy \
    clang-format \
    gdb \
    python3 \
    python3-dev \
    pip \
    && pip3 install \
    dark \
    pytest \
    mlflow \
    toml \
    psutil \
    && echo "export PYTHONPATH=/src/build:$PYTHONPATH" >> $HOME/.bash_profile