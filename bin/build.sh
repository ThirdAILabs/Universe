#!/bin/bash

BASEDIR=$(dirname "$0")

mkdir "$BASEDIR/../build"

cmake -S "$BASEDIR/../" -B "$BASEDIR/../build"

make all -C "$BASEDIR/../build"