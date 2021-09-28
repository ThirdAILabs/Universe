#!/bin/bash

BASEDIR=$(dirname "$0")
./$BASEDIR/tests.sh
./$BASEDIR/get_datasets.sh

export DATE=$(date '+%Y-%m-%d')
target=$BASEDIR/../../logs/$DATE
mkdir $BASEDIR/../../logs/
mkdir $target

export NOW=$(date +"%T")

# We need: Code version, machine information, run time, accuracy, hash seeds
cd $BASEDIR/../build/
echo "<html>" >> "../$target/$NOW.html"
lscpu > "../$target/$NOW.html"
echo "" >> "../$target/$NOW.html"
echo "" >> "../$target/$NOW.html"

echo "Current code version:" >> "../$target/$NOW.html"
git describe --tag >> "../$target/$NOW.html"
echo "" >> "../$target/$NOW.html"

ctest -A >> "../$target/$NOW.html"
echo "</html>" >> "../$target/$NOW.html"
