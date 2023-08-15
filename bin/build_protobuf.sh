#!/bin/bash

# Run this script with sudo.
# This script is to build and install protobuf from source. This is used to install 
# protobuf for building our linux wheels because the package manager, yum, installs 
# an older version which doesn't support some of the features we want.

cd deps/protobuf

# Follows instructions to build from source here:
# https://github.com/protocolbuffers/protobuf/tree/v3.19.3/src
./autogen.sh
./configure
make
make install
ldconfig
