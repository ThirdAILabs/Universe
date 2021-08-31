find . -iname '*.h' -o -iname '*.cc' | xargs clang-format -i -style=file
python3 run-clang-tidy.py