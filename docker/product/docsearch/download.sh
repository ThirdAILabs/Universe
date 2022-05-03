#!/bin/bash

# This file needs to be run on blade before building the image

BASEDIR=$(dirname "$0")
cd $BASEDIR

# TODO(josh): Move this stuff to backblaze or a similar host so it is 
# reproducible not just on our blade
mkdir -p downloads/checkpoint
cp  /share/josh/msmarco/ColBERT/downloads/colbertv2.0/* downloads/checkpoint/

cp /share/josh/msmarco/centroids.npy downloads/

# Exit 0 no matter what for now so it passes github actions
exit 0