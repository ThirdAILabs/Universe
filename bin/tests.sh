#!/bin/bash

BASEDIR=$(dirname "$0")

# Make sure the code is built
"$BASEDIR/build.sh"

BUILDDIR="$BASEDIR/../build"

# Download and unzip svm datasets
SVMDATADIR="$BUILDDIR/utils/tests/dataset/svm" 
echo $SVMDATADIR
curl "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/bibtex.bz2" --output $SVMDATADIR/bibtex.bz2
bzip2 -d $SVMDATADIR/bibtex.bz2

# Run tests with the passed in arguments
cd $BASEDIR/../build/
ctest



