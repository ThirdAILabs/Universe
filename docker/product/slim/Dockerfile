# Ubuntu 20.04 (focal)
# https://hub.docker.com/_/ubuntu/?tab=tags&name=focal
ARG ROOT_CONTAINER=ubuntu:focal
FROM $ROOT_CONTAINER

SHELL ["/bin/bash", "-c"]
ADD Universe.tar.bz2 /Universe
ADD thirdai_smoke_test.py /
USER root
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --yes ; \
    apt-get upgrade --yes ;
RUN apt-get install --yes --no-install-recommends \
    build-essential \
    git \
    curl \
    cmake \
    python3 \
    python3-dev \
    pip
RUN /Universe/bin/build.sh

FROM $ROOT_CONTAINER
COPY --from=0 /Universe/build/thirdai.cpython-* ./
COPY thirdai_smoke_test.py /
RUN RUN apt-get update --yes ; \
    apt-get upgrade --yes ; \
    apt-get install --yes --no-install-recommends \
    python3 \
    pip ; \
    echo "export PYTHONPATH=/" >> ~/.bashrc