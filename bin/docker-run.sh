#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR/../
# Run with --cap-add and --security-opt to be able to trace exectuion, see
# https://stackoverflow.com/questions/19215177/how-to-solve-ptrace-operation-not-permitted-when-trying-to-attach-gdb-to-a-pro

DATADIR="/shared/data"
if [ -d "${DATADIR}" ]; then
    docker run \
      --privileged --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      -it --rm \
      --mount type=bind,source=${DATADIR},target=/data \
      --mount type=bind,source=${PWD},target=/Universe thirdai/universe_dev_build \
      bash
else
    docker run \
      --privileged --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      -it --rm \
      --mount type=bind,source=${PWD},target=/Universe thirdai/universe_dev_build \
      bash
fi
