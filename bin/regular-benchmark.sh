#!/bin/bash

BASEDIR=$(dirname "$0")
./$BASEDIR/build.sh
./$BASEDIR/tests.sh

date=$(date '+%Y-%m-%d')
mkdir /home/logs/date

now=$(date +"%T")
#echo "today: $date"
#echo "Current time : $now"
cd /$BASEDIR/../build/
ctest -A >> "${now}.txt"