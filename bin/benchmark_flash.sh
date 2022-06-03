#!/bin/bash

# This script runs our benchmarks and logs the results to our MlFlow server.

# TODO (vihan): Get this benchmarking working with latest mlflow wrapper

#BASEDIR=$(dirname "$0")
#BENCHMARKING_FOLDER=$BASEDIR/../benchmarks
#
##  --------------- Mag Search ---------------
## Image net embedding search
#IMAGENET_FOLDER=$BENCHMARKING_FOLDER/magsearch/imagenet
#mkdir -p $IMAGENET_FOLDER
#python3 $BENCHMARKING_FOLDER/flash_benchmarks/image_search.py \
#--read_in_entire_dataset \
#> $IMAGENET_FOLDER/stdout 2> $IMAGENET_FOLDER/stderr
#
#DATE=$(date '+%Y-%m-%d')
#LOG_DIR=$BASEDIR/../../logs/$DATE
#mkdir -p $LOG_DIR
#cp -a $BENCHMARKING_FOLDER/* $LOG_DIR
