#!/bin/bash

# This script runs our benchmarks and logs the results to our MlFlow server.

BASEDIR=$(dirname "$0")
BENCHMARKING_FOLDER=$BASEDIR/../benchmarks

#  --------------- Mag Search ---------------
if [ "$RUN_MAGSEARCH" != "n" ]
then
  # Image net embedding search
  IMAGENET_FOLDER=$BENCHMARKING_FOLDER/magsearch/imagenet
  mkdir -p $IMAGENET_FOLDER
  python3 $BASEDIR/../flash/benchmarks/image_search.py \
    --read_in_entire_dataset \
    > $IMAGENET_FOLDER/stdout 2> $IMAGENET_FOLDER/stderr
fi

# --------------- Bolt ---------------
  # Amazon 670k extreme classification
  # AMAZON_FOLDER=$BENCHMARKING_FOLDER/bolt/amazon670k
  # mkdir -p $AMAZON_FOLDER
 python3 $BASEDIR/../benchmarks/bolt_benchmarks/bolt.py $BENCHMARKING_FOLDER/bolt_benchmarks/configs/amzn670k.txt --disable_mlflow

python3 $BASEDIR/../benchmarks/bolt_benchmarks/bolt.py $BENCHMARKING_FOLDER/bolt_benchmarks/configs/amazon_polarity.txt --disable_mlflow
#DATE=$(date '+%Y-%m-%d')
#LOG_DIR=$BASEDIR/../../logs/$DATE
#mkdir -p $LOG_DIR
#cp -a $BENCHMARKING_FOLDER/* $LOG_DIR
