# Run clang-format on all files/directories recursively
find . -iname '*.h' -o -iname '*.cc' | xargs clang-format -i -style=file