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

bin/run_clang_tidy.py --changed_files_only