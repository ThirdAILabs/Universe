#!/bin/bash
# usage ./create_flamegraph.sh <config_identifier>
# Note where you are calling from does not matter. This tool will create a
# flamegraph in the calling directory called <config_identifier>.svg.
# Example command lines:
# ./create_flamegraph.sh mnist_so
# ./benchmarks/bolt_benchmarks/create_flamegraph.sh amzn670k
# You need to have built with debug symbold (Debug or RelWithDebInfo)
# and py-spy needs to be installed (with pip3 install py-spy). You also need to
# be running on linux). There must be a config file
# benchmarks/bolt_benchmarks/configs/<config_identifier>.txt
# See https://www.notion.so/Performance-Profiling-a4ccf598339845a7a2d618e9ad019e88
# for more information

# See https://stackoverflow.com/questions/592620/how-can-i-check-if-a-program-exists-from-a-bash-script
type py-spy >/dev/null 2>&1 || { echo >&2 "This script requires py-spy but it's not installed.  Aborting."; exit 1; }

BASEDIR=$(dirname "$0")
RAW_OUTPUT_LOC=$BASEDIR/raw.txt

# --format raw tells py-spy to print the raw callstacks instead of generating a flamegraph itself, 
# since flamegraph.pl generates a slightly better one. We limit the sampling rate to 20 because
# the default of 100 can start printing error messages saying sampling is falling behind. --nolineno
# turns off line numbers so calls from the same function will always get grouped together. 
# --native enavles profiling of C++ libraries, most importantly our own!
py-spy record --format raw --output $RAW_OUTPUT_LOC --rate 20 --nolineno --native \
    -- python3 $BASEDIR/run_bolt_experiment.py $BASEDIR/configs/$1.txt --disable_mlflow

$BASEDIR/../../deps/flamegraph/flamegraph.pl $RAW_OUTPUT_LOC > $config_identifier.svg

rm $RAW_OUTPUT_LOC