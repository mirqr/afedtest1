#!/bin/bash

p=9
k=10
m=$((p*k))

step=15
for i in $(seq $step $step $m)
do
    echo "Welcome $((i-$step)) $i  times"
    python 02_main.py $((i-$step)) $i 1 &
done
if [ $i -ne $m ]
then
    echo "Welcome $(($m-$step+1)) $m  times"
    python 02_main.py $(($m-$step+1)) $m 1 &
fi


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait



for i in $(seq $step $step $m)
do
    echo "Welcome $((i-$step)) $i  times"
    python 02_main.py $((i-$step)) $i 0 &
done
if [ $i -ne $m ]
then
    echo "Welcome $(($m-$step+1)) $m  times"
    python 02_main.py $(($m-$step+1)) $m 0 &
fi


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait


echo "END"