#!/bin/bash

BASEDIR=$(dirname "$0")
BUILDDIR="$BASEDIR/../build"

# Download and unzip data
SVMDATDIR="$BUILDDIR/utils/tests/dataset/svm" 
wget -P $SVMDATADIR https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/bibtex.bz2
bunzip $SVMDATADIR/bibtex
