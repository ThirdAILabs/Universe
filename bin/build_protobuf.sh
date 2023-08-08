#!/bin/bash

# Run this script with sudo.

git clone https://github.com/protocolbuffers/protobuf --recursive
cd protobuf/

# Newer versions have an issue where abseil and utf8_range don't get linked 
# correctly and it causes link errors unless we manually link them directly. 
git checkout tags/v3.19.3 

# Follows instructions to build from source here:
# https://github.com/protocolbuffers/protobuf/tree/v3.19.3/src
./autogen.sh
./configure
make
make install
ldconfig

# Cleanup after installation.
cd ..
rm -rf protobuf