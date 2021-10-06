#!/bin/bash

BASEDIR=$(dirname "$0")

mkdir -p "$BASEDIR/../build"

cmake -S "$BASEDIR/../" -B "$BASEDIR/../build"

make all -s -C "$BASEDIR/../build"