#!/bin/bash

ROUNDS=10


function run {
    # taskset -c 0-23 ./basic g $1 $ROUNDS $2 1 | grep "F-Graph" | sed "s/##/24/g"

    # taskset -c 0-23,48-71 ./basic g $1 $ROUNDS $2 1 | grep "F-Graph" | sed "s/##/48h/g"

    # taskset -c 0-47 ./basic g $1 $ROUNDS $2 1 | grep "F-Graph" | sed "s/##/48n/g"

    numactl -i all ./basic g $1 $ROUNDS $2 1 | grep "F-Graph" | sed "s/##/96/g"
}
 
# run  /home/ubuntu/graphs/soc-LiveJournal1_sym.adj    0           
# run  /home/ubuntu/graphs/com-orkut.ungraph.adj  1000               
# run  /home/ubuntu/graphs/rmat_ligra.adj       0            
# run  /home/ubuntu/graphs/er_graph.adj                 0      
# run  /home/ubuntu/graphs/twitter.adj            12      
#run  /home/ubuntu/graphs/com-friendster.ungraph.adj            100000      
run  /home/ubuntu/graphs/AGATHA_2015.adj            100000      
 
