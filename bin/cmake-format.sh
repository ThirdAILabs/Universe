BASEDIR=$(dirname "$0")

find "$BASEDIR/../" -type f \( -name 'CMakeLists.txt' -o -name '*.cmake' \) -not -path "*/deps/*" -exec cmake-format -i {} \;