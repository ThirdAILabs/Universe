#!/bin/bash

BASEDIR=$(dirname "$0")
./$BASEDIR/tests.sh
./$BASEDIR/get_datasets.sh

date=$(date '+%Y-%m-%d')
target=$BASEDIR/../../logs/$date
mkdir $BASEDIR/../../logs/
mkdir $target

now=$(date +"%T")

# We need: Code version, machine information, run time, accuracy, hash seeds
cd $BASEDIR/../build/
lscpu > "../$target/$now.txt"
echo "" >> "../$target/$now.txt"
echo "" >> "../$target/$now.txt"

echo "Current code version:" >> "../$target/$now.txt"
git describe --tag >> "../$target/$now.txt"
echo "" >> "../$target/$now.txt"

ctest -A >> "../$target/$now.txt"

