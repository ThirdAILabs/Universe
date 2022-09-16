#!/bin/bash

# This script will build Universe, build the docs, and then use credentials 
# stored on blade to ssh into the T2 Micro docs machine, delete the old docs, 
# and then copy the built docs in their place.

BASEDIR=$(dirname "$0")

cd $BASEDIR/..
bin/build.py -f THIRDAI_BUILD_LICENSE

cd docs
make clean
make html

cd ..

ssh -i /share/keys/DocsServer.pem ubuntu@ec2-3-14-67-7.us-east-2.compute.amazonaws.com "rm -rf /home/ubuntu/html"
scp -r -i /share/keys/DocsServer.pem docs/_build/html ubuntu@ec2-3-14-67-7.us-east-2.compute.amazonaws.com:/home/ubuntu/

