#!/bin/bash
# usage ./create_flamegraph.sh <model_type> <config_identifier>
# model_type is either "engine" or "text". If no match is found, 
# it defaults to "engine"
# Note where you are calling from does not matter. This tool will create a
# flamegraph in the calling directory called <config_identifier>.svg.
# Example command lines:
# ./create_flamegraph.sh engine mnist_so
# ./benchmarks/bolt_benchmarks/create_flamegraph.sh text amzn670k
# You need to have built with debug symbols (Debug or RelWithDebInfo)
# and perf needs to be installed (so you need to
# be running on linux). There must be a config file
# benchmarks/bolt_benchmarks/configs/<config_identifier>.txt
# See https://www.notion.so/Performance-Profiling-a4ccf598339845a7a2d618e9ad019e88
# for more information.

# See https://stackoverflow.com/questions/592620/how-can-i-check-if-a-program-exists-from-a-bash-script
type perf >/dev/null 2>&1 || { echo >&2 "This script requires perf but it's not installed.  Aborting."; exit 1; }

BASEDIR=$(dirname "$0")

MODEL_TYPE="bolt engine"
BENCHMARK_DIR=bolt_benchmarks
SCRIPT=$BENCHMARK_DIR/run_bolt_experiment.py

if [ $1 == "text" ]
then
    MODEL_TYPE="text classifier"
    BENCHMARK_DIR=text_classifier_benchmarks
    SCRIPT=$BENCHMARK_DIR/run_text_classifier_experiment.py
fi

echo "Creating flamegraph for ${MODEL_TYPE} (script: ${SCRIPT})"

perf record -F 99 --call-graph dwarf python3 $SCRIPT $BENCHMARK_DIR/configs/$2.txt --disable_mlflow
perf script > out.perf

$BASEDIR/../deps/flamegraph/stackcollapse-perf.pl out.perf | $BASEDIR/../deps/flamegraph/flamegraph.pl  > $2.svg

rm out.perf
rm perf.data