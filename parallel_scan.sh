#!/bin/bash
make basic CILK=1
BITS=40
echo "  scan_inplace uncompressed,    scan_inplace compressed"
# for N in $(seq 1000000 1000000 10000000) $(seq 20000000 10000000 100000000) $(seq 200000000 100000000 1000000000)
for N in 1000000 10000000 100000000 1000000000
do
./basic scan_inplace $N $BITS | grep , 
done

# echo "  linear uncompressed,    linear compressed"
# for N in $(seq 1000000 1000000 10000000) $(seq 20000000 10000000 100000000) $(seq 200000000 100000000 1000000000)
# do
# ./basic scan_lin $N $BITS | grep , 
# done

# echo "E uncompressed,   E compressed"
# for N in $(seq 1000000 1000000 10000000) $(seq 20000000 10000000 100000000) $(seq 200000000 100000000 1000000000)
# do
# ./basic scan_eyt $N $BITS | grep , 
# done


# echo "B 9 uncompressed, B 9 compressed"
# for N in $(seq 1000000 1000000 10000000) $(seq 20000000 10000000 100000000) $(seq 200000000 100000000 1000000000)
# do
# ./basic scan_B9 $N $BITS | grep , 
# done


# echo "B 17 uncompressed, B 17 compressed"
# for N in $(seq 1000000 1000000 10000000) $(seq 20000000 10000000 100000000) $(seq 200000000 100000000 1000000000)
# do
# ./basic scan_B17 $N $BITS | grep , 
# done

# for N in $(seq 1000000 1000000 10000000) $(seq 20000000 10000000 100000000) $(seq 200000000 100000000 1000000000)
# do
# ./basic scan_btree $N $BITS | tail -n 1 
# done