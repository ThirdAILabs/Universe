#!/bin/bash


for i in $1; do
    scp start_ray_on_nodes.sh $USER@$i:/home/$USER
    ssh $i 'chmod +x start_ray_on_nodes.sh && ./start_ray_on_nodes.sh' $2
    ssh $i 'rm start_ray_on_nodes.sh'
done
