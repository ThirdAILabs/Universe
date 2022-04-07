#!/bin/bash

# Any command failure will abort the script with an error
set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

./build_all_product_dockers.sh $1

REV_TAG=$(git log -1 --pretty=format:%h)

docker images | grep $REV_TAG | while read -r line ; do

  IMAGE_AND_TAG = $($line | tr -s [:space:] : | cut -f1,2 -d':')
  echo "Testing $IMAGE_AND_TAG"

done

# docker run --privileged -t thirdai_slim_release:$REV_TAG /bin/bash -c "pytest ."
# docker run --privileged -t thirdai_jupyter_release:$REV_TAG /bin/bash -c "pytest ."

# docker run --privileged -t thirdai_docsearch_release:$REV_TAG /bin/bash -c "pytest ."
# docker run --privileged -t thirdai_text_classification_release:$REV_TAG /bin/bash -c "pytest ."