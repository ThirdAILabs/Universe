# Ubuntu 20.04 (focal)
# https://hub.docker.com/_/ubuntu/?tab=tags&name=focal
ARG ROOT_CONTAINER=ubuntu:focal

# First stage, only for building the thirdai.so file
FROM $ROOT_CONTAINER

USER root
SHELL ["/bin/bash", "-c"]

ADD Universe.tar.bz2 /Universe
ADD thirdai_smoke_test.py /

RUN apt-get update --yes ; \
    apt-get upgrade --yes ;
RUN DEBIAN_FRONTEND=noninteractive \ 
    apt-get install --yes --no-install-recommends \
    build-essential \
    git \
    curl \
    cmake \
    python3 \
    python3-dev \
    pip
RUN /Universe/bin/build.sh


# Second stage, we only copy the so file from the first stage
FROM $ROOT_CONTAINER

RUN \
    # Add thirdai user with sudo privledges
    useradd -m -s /bin/bash thirdai ; \
    usermod -aG sudo thirdai ; \
    # Install python and pip
    apt-get update --yes ; \
    apt-get upgrade --yes ; \
    apt-get install --yes --no-install-recommends \
    sudo \
    python3 \
    libgomp1 \
    pip ; \
    # Numpy increases image size by 50% but is necessary for our API calls
    pip3 install pytest numpy

# Swtich to to new thirdai user
USER thirdai
WORKDIR /home/thirdai

# Copy in smoke test and thirdai so
COPY thirdai_smoke_test.py .
COPY --from=0 /Universe/build/thirdai.cpython-* .

# Allow us to use so from anywhere
RUN echo "export PYTHONPATH=/" >> ~/.bashrc ;