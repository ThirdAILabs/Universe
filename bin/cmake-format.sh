BASEDIR=$(dirname "$0")

find "$BASEDIR/../" -type f \( -iname 'CMakeLists.txt' -o -iname '*.cmake' \) -not -path "*/deps/*" -exec cmake-format -i {} \;