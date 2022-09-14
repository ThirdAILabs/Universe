#!/bin/bash
# First usage ./create_flamegraph.sh <model_flag> <config_identifier>
# <model_flag> is either "-b" to create a flamegraph for the bolt engine
# or "-s" for the sequential classifier.
#
# Alternate usage ./create_flamegraph.sh <python_script> <flamegraph_name>
#
# Note where you are calling from does not matter. 
# If this tool is called with a model_flag, it will create a flamegraph in 
# the calling directory called <config_identifier>.svg. 
# Otherwise, it will create a flamegraph in the calling directory called
# <flamegraph_name>.svg.
# Example command lines:
# ./create_flamegraph.sh -b mnist_so
# ./benchmarks/create_flamegraph.sh -b amzn670k
# ./benchmarks/create_flamegraph.sh -s amazon_games
# ./create_flamegraph.sh "bolt.py bolt_configs/amzn670k.txt --disable_mlflow" my_svg
# You need to have built with debug symbols (Debug or RelWithDebInfo)
# and perf needs to be installed (so you need to
# be running on linux). There must be a config file
# benchmarks/bolt_configs/<config_identifier>.txt or 
# benchmarks/sequential_classifier_configs/<config_identifier>.txt
# See https://www.notion.so/Performance-Profiling-a4ccf598339845a7a2d618e9ad019e88
# for more information.

# See https://stackoverflow.com/questions/592620/how-can-i-check-if-a-program-exists-from-a-bash-script
type perf >/dev/null 2>&1 || { echo >&2 "This script requires perf but it's not installed.  Aborting."; exit 1; }

BASEDIR=$(dirname "$0")

case "$1" in
    "-b") 
        echo "Creating flamegraph for bolt engine" 
        SCRIPT="$BASEDIR/bolt.py $BASEDIR/bolt_configs/$2.txt --disable_mlflow"
        ;;
      
    "-s") 
        echo "Creating flamegraph for sequential classifier" 
        SCRIPT="$BASEDIR/sequential_classifier.py $BASEDIR/sequential_classifier_configs/$2.txt --disable_mlflow"
        ;;
    
    #default
    *)
        echo "Creating flamegraph for python script '$1'"
        SCRIPT=$1

esac


perf record -F 99 --call-graph dwarf python3 $SCRIPT
perf script > out.perf

$BASEDIR/../deps/flamegraph/stackcollapse-perf.pl out.perf | $BASEDIR/../deps/flamegraph/flamegraph.pl  > $2.svg

rm out.perf
rm perf.data