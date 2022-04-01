BASEDIR=$(dirname "$0")

cd $BASEDIR/../build

rm CMakeCache.txt

cd _deps

find . -name "*CMakeCache.txt" -type f -delete
