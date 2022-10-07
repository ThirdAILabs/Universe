#!/bin/bash

export USE_CCACHE=1 

# ThirdAI builds from my local machine are license-check-free for quick
# prototyping abilities.
export THIRDAI_FEATURE_FLAGS=THIRDAI_EXPOSE_ALL
export THIRDAI_DATASET_PATH="$HOME/data"
export THIRDAI_BUILD_MODE=RelWithDebInfo


BASEDIR=$(dirname -- "${BASH_SOURCE[0]}")
THIRDAI_SOURCE_DIR="$BASEDIR/.."

function thirdai-build {
    set -x;
    (cd $THIRDAI_SOURCE_DIR && (THIRDAI_BUILD_IDENTIFIER=$(git rev-parse --short HEAD) THIRDAI_BUILD_MODE=RelWithDebInfo python3 setup.py bdist_wheel 2>&1 | tee compile-verbose.log))
    set +x;
}

function thirdai-install {
    WHEEL_TAG="$(cat $THIRDAI_SOURCE_DIR/thirdai.version)+$(git -C $THIRDAI_SOURCE_DIR rev-parse --short HEAD)"
    set -x;
    python3 -m pip install --force-reinstall --upgrade \
        $THIRDAI_SOURCE_DIR/dist/thirdai-${WHEEL_TAG}-cp310-cp310-macosx_12_0_arm64.whl
    set +x;
}
