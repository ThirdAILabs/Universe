#!/bin/bash

BASEDIR=$(dirname "$0")
BUILDDIR="$BASEDIR/../build"

# Download and unzip data
SVMDATADIR="$BUILDDIR/utils/tests/dataset/svm" 
if [ ! -d "$SVMDATADIR" ]; then
    curl "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/bibtex.bz2" --output $SVMDATADIR/bibtex.bz2
    bzip2 -d $SVMDATADIR/bibtex.bz2
    echo "Downloaded $SVMDATADIR/bibtex.bz2"
fi