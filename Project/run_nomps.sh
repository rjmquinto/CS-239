#!/bin/bash

LOG_FOLDER=results/nomps

sudo ./mps-stop.sh

rm -rf $LOG_FOLDER
mkdir -p $LOG_FOLDER


nvidia-smi stats -d gpuUtil > $LOG_FOLDER/gpuUtil.csv &
log_pid=$!

echo "START $(date)" > $LOG_FOLDER/timestamp

for i in {1..5}; do
	/usr/bin/time -f "%e" -o $LOG_FOLDER/lstm$i.log ./main_lstm > /dev/null &
done

for i in {1..5}; do
	/usr/bin/time -f "%e" -o $LOG_FOLDER/relu$i.log ./main_relu > /dev/null &
done



while [ -n "$(ps -e | grep main_lstm)" -o -n "$(ps -e | grep main_relu)" ]
do
	sleep 1
done

sleep 1
kill $log_pid

echo "END $(date)" >> $LOG_FOLDER/timestamp
