#!/bin/bash
# Usage ./create_flamegraph.sh <python_script> <flamegraph_name>
#
# Example command line:
# ./create_flamegraph.sh "bolt.py bolt_configs/amzn670k.txt --disable_mlflow" my_svg
# You need to have built with debug symbols (Debug or RelWithDebInfo)
# and perf needs to be installed (so you need to
# be running on linux). 
# See https://www.notion.so/Performance-Profiling-a4ccf598339845a7a2d618e9ad019e88
# for more information.

# See https://stackoverflow.com/questions/592620/how-can-i-check-if-a-program-exists-from-a-bash-script
type perf >/dev/null 2>&1 || { echo >&2 "This script requires perf but it's not installed.  Aborting."; exit 1; }

BASEDIR=$(dirname "$0")


perf record -F 99 --call-graph dwarf python3 $1
perf script > out.perf

$BASEDIR/../deps/flamegraph/stackcollapse-perf.pl out.perf | $BASEDIR/../deps/flamegraph/flamegraph.pl  > $2.svg

rm out.perf
rm perf.data