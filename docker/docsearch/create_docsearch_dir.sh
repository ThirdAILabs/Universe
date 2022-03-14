#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

CURRENT_BRANCH=$(git branch | sed -n -e 's/^\* \(.*\)/\1/p')
cd ../../
git archive $CURRENT_BRANCH | bzip2 > $BASEDIR/Universe.tar.bz2