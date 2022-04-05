#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR


wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz -P ColBERT/downloads/
tar -xvzf downloads/colbertv2.0.tar.gz -C downloads/

cp /share/josh/msmarco/centroids.npy 

cp artifact.metadata downloads/colbertv2.0/