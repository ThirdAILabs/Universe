#!/bin/bash

# Any command failure will abort the script with an error
set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

./build_all_product_dockers.sh

REV_TAG=$(git log -1 --pretty=format:%h)

docker run --user 1000:100 -t thirdai_slim_release:$REV_TAG /bin/bash -c "pytest ."
docker run --user 1000:100 -t thirdai_jupyter_release:$REV_TAG /bin/bash -c "pytest ."

docker run --user 1000:100 -t thirdai_docsearch_release:$REV_TAG /bin/bash -c "pytest ."