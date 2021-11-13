#!/bin/bash

curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 --output mnist.bz2
curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 --output mnist.t.bz2

bzip2 -d mnist.bz2
bzip2 -d mnist.t.bz2

BASEDIR=$(dirname "$0")
python3 $BASEDIR/../bolt/benchmarks/runner.py mnist_sparse_output --enable_checks --train ./mnist --test ./mnist.t --runs 5
python3 $BASEDIR/../bolt/benchmarks/runner.py mnist_sparse_hidden --enable_checks --train ./mnist --test ./mnist.t --runs 5