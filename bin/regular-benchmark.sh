#!/bin/bash

BASEDIR=$(dirname "$0")
./$BASEDIR/tests.sh
./$BASEDIR/get_datasets.sh

date=$(date '+%Y-%m-%d')
target=$BASEDIR/../../logs/$date
mkdir $BASEDIR/../../logs/
mkdir $target

now=$(date +"%T")

cd $BASEDIR/../build/
ctest -A > "../$target/$now.txt"

