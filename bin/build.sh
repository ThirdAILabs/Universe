#!/bin/bash
# Usage ./bin/build.sh [make_target] [build mode] [num_jobs]
# The first positional argument is the make target. If omitted, make will use 'all', but
# you can pass in a specific one if you just want to make one target. e.g.
# run "bin/build.sh clean" if you want to delete objects and executables.
# The second positional argument is whether to use debug or build mode. If ommited it  
# will automatically build in release mode.
# The third positional argument is the number of jobs you wish to run
# make with. If ommited, make will automatically run make with
# the number of jobs = 1.5 * the number of threads on your machine. e.g. to
# run a single thread build of all targets run "bin/build.sh all 1".

BASEDIR=$(dirname "$0")

if [ -z "$1" ] 
then
  TARGET="all"
else
  TARGET="$1"
fi

if [ -z "$2" ] 
then 
  BUILD_MODE=release
else 
  BUILD_MODE=$2
fi

if [ -z "$3" ] 
then
  NUM_AVAILABLE_THREADS=$(getconf _NPROCESSORS_ONLN)
  # Per below, rule of thumb is 1.5 times the number of available threads
  # https://stackoverflow.com/questions/414714/compiling-with-g-using-multiple-cores
  NUM_JOBS=$((3*NUM_AVAILABLE_THREADS/2))
else
  NUM_JOBS="$3"
fi

mkdir -p "$BASEDIR/../build"

cd "$BASEDIR/../build"

cmake .. -DPYTHON_EXECUTABLE=$(which python3) -DCMAKE_BUILD_TYPE=$BUILD_MODE

make $TARGET -s -j$NUM_JOBS