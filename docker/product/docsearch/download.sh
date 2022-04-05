#!/bin/bash

# This file needs to be run on blade before building the image

BASEDIR=$(dirname "$0")
cd $BASEDIR

# TODO(josh): Move this stuff to backblaze so it is reproducible not just on our blade
cp  /share/josh/msmarco/ColBERT/downloads/colbertv2.0.tar.gz ColBERT/downloads/
tar -xvzf ColBERT/downloads/colbertv2.0.tar.gz -C ColBERT/downloads/

cp /share/josh/msmarco/centroids.npy ColBERT/downloads

# Override the existing artifact metadata
cp ColBERT/artifact.metadata ColBERT/downloads/colbertv2.0/