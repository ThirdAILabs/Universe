#!/bin/bash

./generate_compile_commands.sh
find . -type f -not -path "./build/*" -iname '*.h' -o -iname '*.cc' | xargs clang-format -i -style=file
python3 run-clang-tidy.py