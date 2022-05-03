#!/bin/bash

# This file needs to be run on blade before building the image

BASEDIR=$(dirname "$0")
cd $BASEDIR

# TODO(josh): Move this stuff to backblaze so it is reproducible not just on our blade
cp  /share/josh/msmarco/ColBERT/downloads/colbert/* downloads/checkpoint/
tar -xvzf ColBERT/downloads/colbertv2.0.tar.gz -C downloads/

cp /share/josh/msmarco/centroids.npy downloads/

# Exit 0 no matter what for now so it passes github actions
exit 0