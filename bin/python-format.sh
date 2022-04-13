#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
black . --exclude deps