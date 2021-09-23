#!/bin/bash

BASEDIR=$(dirname "$0")
./$BASEDIR/tests.sh

date=$(date '+%Y-%m-%d')
target=$BASEDIR/../../logs/$date
mkdir $BASEDIR/../../logs/
mkdir $target

now=$(date +"%T")
#echo "today: $date"
#echo "Current time : $now"
cd $BASEDIR/../build/
ctest -A > "../$target/$now.txt"
#ctest -A > "$now.txt"
