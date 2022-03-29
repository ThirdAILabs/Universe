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

num_failed=0

# Run clang tidy on all headers and source file, ignoring the build and deps directories
# Note this will break if a filename has a new line in it
find ./ -type f \( -iname \*.h -o -iname \*.cc \) \
  -not -path "./deps/*" -not -path "./build/*" \
    | while read fname; do 
      echo $fname 
      clang-tidy $fname || ((num_failed++))
    done

exit num_failed