#!/bin/bash

# Any command failure will abort the script with an error
set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

./build_all_product_dockers.sh

docker run --user 1000:100 -t base_product /bin/bash -c "pytest ."
docker run --user 1000:100 -t docsearch /bin/bash -c "pytest ."