#!/bin/bash

BASEDIR=$(dirname "$0")

# Make sure compile commands database is freshly created
$BASEDIR/generate_compile_commands.sh

# Check all files have pragma once
NEED_PRAGMA=$(grep -L -r "pragma once" $BASEDIR/../. --include \*.h --exclude-dir=build --exclude-dir=deps)
if [ -n "$NEED_PRAGMA" ]
then
  echo "The following files need pragma once:"
  echo $NEED_PRAGMA
  exit 1
fi

# cd to Universe
cd $BASEDIR/../

# See https://stackoverflow.com/questions/1000370/add-collect-exit-codes-in-bash
declare -i NUM_FAILED=0

# TODO(josh): Do we want to check headers too? Makes it take longer but 
# could easily add -iname \*.h -o
# Run clang tidy on all headers and source file, ignoring the build and deps directories
# Note this will break if a filename has a new line in it
# See http://mywiki.wooledge.org/BashFAQ/024 for why we have to do return the
# NUM_FAILED within the pipe (it's a subshell)
find . -type f -name "*.cc" \
  -not -path "./deps/*" -not -path "./build/*" | 
{ 
  while read fname; do 
    echo $fname 
    clang-tidy $fname -- --std=c++17
    NUM_FAILED+=$?
  done

  exit $NUM_FAILED
}