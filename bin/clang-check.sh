#!/bin/bash

BASEDIR=$(dirname "$0")

"$BASEDIR"/generate_compile_commands.sh
find "$BASEDIR/../" -type f \( -iname '*.h' -o -iname '*.cc' \) -not -path "*/build/*" | xargs clang-format -i -style=file
"$BASEDIR"/tidy-check.sh