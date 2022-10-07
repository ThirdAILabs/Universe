#!/bin/bash

source /opt/intel/oneapi/vtune/latest/env/vars.sh

set -x;


VERSION=$(python3 -c "import thirdai; print(thirdai.__version__)")
RESULTS_BASE=perf/$VERSION

mkdir -p $RESULTS_BASE;

vtune -collect hotspots -result-dir $RESULTS_BASE/hotspots python3 benchmarks/bolt.py benchmarks/bolt_configs/criteo_dlrm.txt --disable_mlflow --log-to-stderr
