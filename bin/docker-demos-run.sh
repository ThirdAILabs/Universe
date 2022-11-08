#!/bin/bash

BASEDIR=$(dirname "$0")

cd $BASEDIR/../

docker build -t thirdai/run_demos_build -f docker/demos/Dockerfile .

KAGGLE_CREDENTIALS_DIR="~/.kaggle"
docker run --mount type=bind,source=${DATADIR},target=/data thirdai/run_demos_build bash  -c "python3 run_demo_notebooks.py"