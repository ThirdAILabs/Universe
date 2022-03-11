#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR
BASEDIR=$(pwd)

cd ../../
git archive main | bzip2 > $BASEDIR/Universe.tar.bz2