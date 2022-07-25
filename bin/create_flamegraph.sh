#!/bin/bash
# usage ./create_flamegraph.sh program_to_profile arg1 arg2...

# TODO(anyone): This just works on linux. Add support for mac?
# See https://stackoverflow.com/questions/592620/how-can-i-check-if-a-program-exists-from-a-bash-script
type foo >/dev/null 2>&1 || { echo >&2 "This script requires perf but it's not installed.  Aborting."; exit 1; }

perf record -F 99 --call-graph dwarf -o perf.out "$@"

BASEDIR=$(dirname "$0")
$BASEDIR/../deps/flamegraph/stackcollapse.pl perf.out | $BASEDIR/../deps/flamegraph/flamegraph.pl  > flamegraph.svg

rm perf.out