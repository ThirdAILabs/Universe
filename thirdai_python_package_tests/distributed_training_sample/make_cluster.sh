#!/bin/bash


for i in $1; do
    scp start_ray_on_nodes.sh $USER@192.168.1.$i:/home/$USER
    ssh 192.168.1.$i './start_ray_on_nodes.sh'
    ssh 192.168.1.$i 'rm start_ray_on_nodes.sh'
done
