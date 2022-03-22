#!/bin/bash

# Any command failure will abort the script with an error
set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

./build_all_product_dockers.sh

docker run -t base_product pytest
docker run -t docsearch pytest