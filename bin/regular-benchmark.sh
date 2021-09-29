#!/bin/bash

BASEDIR=$(dirname "$0")
./$BASEDIR/build.sh
./$BASEDIR/get_datasets.sh

export DATE=$(date '+%Y-%m-%d')
target=$BASEDIR/../../logs/$DATE
mkdir $BASEDIR/../../logs/
mkdir $target
export NOW=$(date +"%T")

# We need: Code version, machine information, run time, accuracy, hash seeds
cd $BASEDIR/../build/
LOGFILE="../$target/$NOW.txt"
lscpu > $LOGFILE
echo "---------------------------------------" >> $LOGFILE
echo "Current code version:" >> $LOGFILE
git describe --tag >> $LOGFILE
echo "---------- Unit Test Results ----------" >> $LOGFILE
ctest -A >> $LOGFILE

sh ./$BASEDIR/bolt_mnist_test.sh