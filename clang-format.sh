find . -iname '*.h' -o -iname '*.cc' | xargs clang-format -i -style=file
find . -iname '*.h' -o -iname '*.cc' | xargs clang-tidy