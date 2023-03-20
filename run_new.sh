#!/bin/bash
echo "Starting server"


#python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data() "

#dataset=$1
#address=$2

num_cl=$#
#echo $num_clients
let "num_cl-=4"
#echo $num_clients

python server.py --address=$2 --num_clients=$num_cl &
sleep 7  # Sleep for 3s to give the server enough time to start


for i in "${@:5}"
do
    echo "Starting client $i"
    python client.py --data_name=$1 --address=$2 --partition=${i} --num_clients=$3 --out_fold=$4 &
done



# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

echo "END"