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
type perf >/dev/null 2>&1 || { echo >&2 "This script requires perf but it's not installed.  Aborting."; exit 1; }

BASEDIR=$(dirname "$0")

perf record -F 99 --call-graph dwarf python3 $BASEDIR/run_bolt_experiment.py $BASEDIR/configs/$1.txt --disable_mlflow
perf script > out.perf

$BASEDIR/../../deps/flamegraph/stackcollapse-perf.pl out.perf | $BASEDIR/../../deps/flamegraph/flamegraph.pl  > $1.svg

rm out.perf
rm perf.data