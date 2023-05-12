#!/bin/sh

python -m venv old_thirdai
source old_thirdai/bin/activate
bin/build.py --extras test
pip uninstall -y thirdai