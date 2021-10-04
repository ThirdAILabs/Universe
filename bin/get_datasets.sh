#!/bin/bash

BASEDIR=$(dirname "$0")
BUILDDIR="$BASEDIR/../build"

# Download and unzip data
SVMDATADIR="$BUILDDIR/utils/tests/dataset/svm" 
echo $SVMDATADIR
curl "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/bibtex.bz2" --output $SVMDATADIR/bibtex.bz2
bzip2 -d $SVMDATADIR/bibtex.bz2
