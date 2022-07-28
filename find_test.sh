#!/bin/bash
make CILK=1 basic
NUM_SEARCH=10000000
for N in $(seq 1000000 1000000 10000000) $(seq 20000000 10000000 100000000) $(seq 200000000 100000000 1000000000)
do
./basic find $N $NUM_SEARCH  | grep time |awk -F " " '{print $3 $6}' |xargs -n14| sed 's/ /,/g' | sed 's/,/, /g' 
done
