#!/bin/bash
# Usage ./bin/build.sh
# Will automatically run make with 1.5 * the number of threads on your machine
# If you want to run with less jobs, or single threaded, pass a single 
# positional command line parameter denoting the number of jobs you wish to
# run make with.

BASEDIR=$(dirname "$0")

if [ -z "$1" ] 
then
  NUM_AVAILABLE_THREADS=$(getconf _NPROCESSORS_ONLN)
  # Per below, rule of thumb is 1.5 times the number of available threads
  # https://stackoverflow.com/questions/414714/compiling-with-g-using-multiple-cores
  NUM_JOBS=$((3*NUM_AVAILABLE_THREADS/2))
else
  NUM_JOBS="$1"
fi

mkdir -p "$BASEDIR/../build"

cmake -S "$BASEDIR/../" -B "$BASEDIR/../build"

make all -C "$BASEDIR/../build" -j$NUM_JOBS