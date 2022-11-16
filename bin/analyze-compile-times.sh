#!/bin/bash

# This script runs clang-build-analyser on thirdai package's python builds.
# Usage: bash <path-to-this-script> <path-to-clang-build-analyzer> <build-folder-where-artifacts-present>

# The script requires ClangBuildAnalyzer, clang to enable the CompileTrace builds.

# Outputs are written in a compile-analysis folder.
# analysis.txt holds the summary, while capture.txt is not human-readable and stores the raw data.


CLANG_BUILD_ANALYSER=$1
ARTIFACTS_FOLDER=$2
BUILD_MODE="CompileAnalysis"
BASEDIR=$(dirname -- $0)
THIRDAI_SOURCE_DIR=$(realpath $BASEDIR/..)

set -x;

$CLANG_BUILD_ANALYSER --start $ARTIFACTS_FOLDER

(cd $THIRDAI_SOURCE_DIR && \
    (
        THIRDAI_BUILD_IDENTIFIER=$(git rev-parse --short HEAD) 
        THIRDAI_BUILD_MODE=$BUILD_MODE CC=clang CXX=clang++ python3 setup.py bdist_wheel 2>&1 \
            | tee compile-verbose.log))

OUTPUTS_FOLDER="compile-analysis"
mkdir -p $OUTPUTS_FOLDER

$CLANG_BUILD_ANALYSER --stop $ARTIFACTS_FOLDER $OUTPUTS_FOLDER/capture.txt

$CLANG_BUILD_ANALYSER --analyze $OUTPUTS_FOLDER/capture.txt > $OUTPUTS_FOLDER/analysis.txt



