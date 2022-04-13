BASEDIR=$(dirname "$0")

find "$BASEDIR/../" -type f \( -iname '*.h' -o -iname '*.cc' \) -not -path "*/deps/*" | xargs clang-format -i -style=file