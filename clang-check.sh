#!/bin/bash

# ./generate_compile_commands.sh
find . -iname '*.h' -o -iname '*.cc' | xargs clang-format -i -style=file
python3 run-clang-tidy.py