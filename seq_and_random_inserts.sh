#!/bin/bash
make basic 
# echo "  linear uncompressed,    E uncompressed,   B 9 uncompressed, B 17 uncompressed, tlx btree"
# for N in $(seq 100000 100000 1000000) $(seq 2000000 1000000 10000000) $(seq 20000000 10000000 100000000) $(seq 200000000 100000000 1000000000)
# do
# ./basic single_alt $N 1 100  |awk -F "," '{print $2}'  |sed 'N; s/\n/ /g' | sed 's/ \t/,/g' | sed 's/^, //g'
# done

N=10000000
./basic single_alt $N 1 999 |awk -F "," '{print $2}'  |sed 'N; s/\n/ /g' | sed 's/ \t/,/g' | sed 's/^, //g'
./basic single_alt $N 1 99 |awk -F "," '{print $2}'  |sed 'N; s/\n/ /g' | sed 's/ \t/,/g' | sed 's/^, //g'
./basic single_alt $N 2 98 |awk -F "," '{print $2}'  |sed 'N; s/\n/ /g' | sed 's/ \t/,/g' | sed 's/^, //g'
./basic single_alt $N 3 97 |awk -F "," '{print $2}'  |sed 'N; s/\n/ /g' | sed 's/ \t/,/g' | sed 's/^, //g'
./basic single_alt $N 4 96 |awk -F "," '{print $2}'  |sed 'N; s/\n/ /g' | sed 's/ \t/,/g' | sed 's/^, //g'
./basic single_alt $N 5 95 |awk -F "," '{print $2}'  |sed 'N; s/\n/ /g' | sed 's/ \t/,/g' | sed 's/^, //g'
./basic single_alt $N 6 94 |awk -F "," '{print $2}'  |sed 'N; s/\n/ /g' | sed 's/ \t/,/g' | sed 's/^, //g'
./basic single_alt $N 7 93 |awk -F "," '{print $2}'  |sed 'N; s/\n/ /g' | sed 's/ \t/,/g' | sed 's/^, //g'
./basic single_alt $N 8 92 |awk -F "," '{print $2}'  |sed 'N; s/\n/ /g' | sed 's/ \t/,/g' | sed 's/^, //g'
./basic single_alt $N 9 91 |awk -F "," '{print $2}'  |sed 'N; s/\n/ /g' | sed 's/ \t/,/g' | sed 's/^, //g'
./basic single_alt $N 10 90|awk -F "," '{print $2}'  |sed 'N; s/\n/ /g' | sed 's/ \t/,/g' | sed 's/^, //g' 