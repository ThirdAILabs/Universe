#!/bin/bash


for i in $1; do
    scp stop_ray_on_nodes.sh $USER@$i:/home/$USER
    ssh $i 'chmod +x stop_ray_on_nodes.sh && ./stop_ray_on_nodes.sh'
    ssh $i 'rm stop_ray_on_nodes.sh'
done
