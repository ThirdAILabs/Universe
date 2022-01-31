BASEDIR=$(dirname "$0")

find "$BASEDIR/../" -type f \( -iname '*.h' -o -iname '*.cc' \) -not -path "*/build/*" | xargs clang-format -i -style=file