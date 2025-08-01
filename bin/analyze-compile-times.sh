#!/bin/bash

# This script runs clang-build-analyser on thirdai package's python builds.
# Usage: bash <path-to-this-script> <path-to-clang-build-analyzer-executable> <build-folder-where-artifacts-present>

# For most configurations, the build-folder where artifacts are present is
# $THIRDAI_SOURCE_DIR/build (including setup.py runs).

# The script requires ClangBuildAnalyzer, clang to enable the CompileTrace builds.

# You may build ClangBuildAnalyzer locally from the instructions at
# https://github.com/aras-p/ClangBuildAnalyzer#building-it 

# Outputs are written in a compile-analysis folder.
# analysis.txt holds the summary, while capture.txt is not human-readable and stores the raw data.


CLANG_BUILD_ANALYSER=$1
ARTIFACTS_FOLDER=$2
BASEDIR=$(dirname -- $0)
THIRDAI_SOURCE_DIR=$(realpath $BASEDIR/..)

set -x;

$CLANG_BUILD_ANALYSER --start $ARTIFACTS_FOLDER

(cd $THIRDAI_SOURCE_DIR &&                      \
    (THIRDAI_FEATURE_FLAGS="THIRDAI_EXPOSE_ALL" \
     THIRDAI_BUILD_MODE="CompileAnalysis"       \
     CC=clang CXX=clang++                       \
     python3 setup.py bdist_wheel 2>&1))

OUTPUTS_FOLDER="compile-analysis"
mkdir -p $OUTPUTS_FOLDER

$CLANG_BUILD_ANALYSER --stop $ARTIFACTS_FOLDER $OUTPUTS_FOLDER/capture.bin

$CLANG_BUILD_ANALYSER --analyze $OUTPUTS_FOLDER/capture.bin > $OUTPUTS_FOLDER/analysis.txt



