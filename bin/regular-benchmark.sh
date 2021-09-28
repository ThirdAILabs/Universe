#!/bin/bash

./tests.sh
./get_datasets.sh

export DATE=$(date '+%Y-%m-%d')
target= ../../logs/$DATE
mkdir ../../logs/
mkdir $target

export NOW=$(date +"%T")

# We need: Code version, machine information, run time, accuracy, hash seeds
cd ../build/
echo "<html>" >> "$target/$NOW.html"
lscpu > "$target/$NOW.html"
echo "" >> "$target/$NOW.html"
echo "" >> "$target/$NOW.html"

echo "Current code version:" >> "$target/$NOW.html"
git describe --tag >> "$target/$NOW.html"
echo "" >> "$target/$NOW.html"

ctest -A >> "$target/$NOW.html"
echo "</html>" >> "$target/$NOW.html"
