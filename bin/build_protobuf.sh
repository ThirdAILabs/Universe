#!/bin/bash

# Run this script with sudo

git clone https://github.com/protocolbuffers/protobuf --recursive
cd protobuf/
mkdir build
cd build
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_STANDARD=14 ..
cmake --build . --parallel 20
cmake --install .
