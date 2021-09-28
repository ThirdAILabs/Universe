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
echo "<html>" >> "../$target/$now.html"
lscpu > "../$target/$now.html"
echo "" >> "../$target/$now.html"
echo "" >> "../$target/$now.html"

echo "Current code version:" >> "../$target/$now.html"
git describe --tag >> "../$target/$now.html"
echo "" >> "../$target/$now.html"

ctest -A >> "../$target/$now.html"
echo "</html>" >> "../$target/$now.html"
