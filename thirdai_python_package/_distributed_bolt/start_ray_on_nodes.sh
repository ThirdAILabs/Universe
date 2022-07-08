#!/bin/bash

pip3 install --no-cache-dir 'ray[default]'
export PATH=$PATH:/home/$USER/.local/bin
ray stop 
ray start --address='192.168.1.2:6379'