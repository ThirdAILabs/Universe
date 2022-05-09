#!/bin/bash

# Any command failure will abort the script with an error
set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

REV_TAG=$(git log -1 --pretty=format:%h)

# Loop through all Docker images ending with REV_TAG (we assume that all valid
# images to test end with this)
docker images | grep $REV_TAG | while read -r line ; do

  # Parse the docker images output by replacing the spaces with colons and 
  # removing everything after the last colon
  IMAGE_AND_TAG=$(echo $line | tr -s [:space:] : | cut -f1,2 -d':')
  echo "Testing $IMAGE_AND_TAG"

  # Run tests on the image
  docker run --privileged -t $IMAGE_AND_TAG /bin/bash -c "pytest ."

done
