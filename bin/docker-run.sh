#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR/../
# Run with --cap-add and --security-opt to be able to trace exectuion, see
# https://stackoverflow.com/questions/19215177/how-to-solve-ptrace-operation-not-permitted-when-trying-to-attach-gdb-to-a-pro

# Mounting the aws credentials is a temporary workaround until we get IAM roles 
# working on AWS. It mounts the AWS credentials present on the home machine to 
# the home directory of the docker image.
DATADIR="/share/data"
if [ -d "${DATADIR}" ]; then
    docker run \
      --privileged --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      -it --rm \
      -v $HOME/.aws/credentials:/root/.aws/credentials:ro \
      --mount type=bind,source=${DATADIR},target=/data \
      --mount type=bind,source=${PWD},target=/Universe thirdai/universe_dev_build \
      bash
else
    docker run \
      --privileged --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      -it --rm \
      -v $HOME/.aws/credentials:/root/.aws/credentials:ro \
      --mount type=bind,source=${PWD},target=/Universe thirdai/universe_dev_build \
      bash
fi
