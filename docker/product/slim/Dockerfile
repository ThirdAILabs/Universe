# Ubuntu 20.04 (focal)
# https://hub.docker.com/_/ubuntu/?tab=tags&name=focal
ARG ROOT_CONTAINER=ubuntu:focal

# First stage, only for building the thirdai.so file
FROM $ROOT_CONTAINER

USER root
SHELL ["/bin/bash", "-c"]

COPY Universe.tar.bz2 /Universe.tar.bz2
RUN tar -xf /Universe.tar.bz2
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
RUN /Universe/bin/build.py -f THIRDAI_CHECK_LICENSE THIRDAI_BUILD_LICENSE


# Second stage, we only copy the so file from the first stage
FROM $ROOT_CONTAINER

RUN \
    # Add thirdai user with sudo privledges
    # See https://stackoverflow.com/questions/25845538/how-to-use-sudo-inside-a-docker-container
    apt-get update \
    && apt-get install -y sudo ; \
    adduser --disabled-password --gecos '' thirdai ;\
    adduser thirdai sudo ;\
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers ;\
    # Install python and pip
    apt-get update --yes ; \
    apt-get upgrade --yes ; \
    apt-get install --yes --no-install-recommends \
    sudo \
    wget \
    python3 \
    libgomp1 \
    pip ; \
    # Numpy increases image size by 50% but is necessary for our API calls
    pip3 install pytest numpy

# Swtich to to new thirdai user
USER thirdai
WORKDIR /home/thirdai

# Checks if sudo works for the thirdai user and updates packages
RUN sudo apt-get update 

# Copy in smoke test and thirdai so
COPY thirdai_smoke_test.py .
COPY --from=0 /Universe/build/thirdai.cpython-* .

# Copy in license files. Use *.serialized to handle case where we want to build
# the docker initially with no license.
COPY license-public-key.der /keys/
COPY *.serialized /licenses/

# Allow us to use so from anywhere
ENV PYTHONPATH="/home/thirdai:${PYTHONPATH}"

# Export path for if the ThirdAI user installs packages locally
ENV PATH="/home/thirdai/.local/bin:${PATH}"
