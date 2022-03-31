#!/bin/bash

# Any command failure will abort the script with an error
set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

./build_all_product_dockers.sh $1

REV_TAG=$(git log -1 --pretty=format:%h)

docker run --privileged -t thirdai_slim_release:$REV_TAG /bin/bash -c "pytest ."
docker run --privileged -t thirdai_jupyter_release:$REV_TAG /bin/bash -c "pytest ."

docker run --privileged -t thirdai_docsearch_release:$REV_TAG /bin/bash -c "pytest ."
docker run --privileged -t thirdai_text_classification_release:$REV_TAG /bin/bash -c "pytest ."