#!/bin/bash
make CYCLE_TIMER=1 ENABLE_TRACE_TIMER=1 basic -B
echo "  total,    find,   modify,   rebalence"
for N in $(seq 100000 100000 1000000) $(seq 2000000 1000000 10000000) $(seq 20000000 10000000 100000000)
do
./basic single $N > del; tail -n 4 del | tac |awk -F "," '{print $2}' |sed 'N;N;N; s/\n/ /g' | sed 's/  /, /g'
done
