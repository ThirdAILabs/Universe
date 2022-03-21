#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR/../
docker run -it --rm --name=UniverseBuild --mount type=bind,source=${PWD},target=/Universe thirdai/universe_dev_build bash