#!/bin/bash
make basic 
echo "  linear uncompressed,    linear compressed,   E uncompressed,   E compressed,  B 9 uncompressed, B 9 compressed, B 17 uncompressed, B 17 compressed, tlx btree"
for N in $(seq 100000 100000 1000000) $(seq 2000000 1000000 10000000) $(seq 20000000 10000000 100000000) $(seq 200000000 100000000 1000000000)
do
./basic single_seq $N  |awk -F "," '{print $2}'  |sed 'N;N;N;N;N;N;N;N; s/\n/ /g' | sed 's/ \t/,/g' | sed 's/^, //g'
done
