#!/bin/bash

LOG_FOLDER=results/mps

sudo ./mps-start.sh

rm -rf $LOG_FOLDER
mkdir -p $LOG_FOLDER


nvidia-smi stats -d gpuUtil > $LOG_FOLDER/gpuUtil.csv &
log_pid=$!

echo "START $(date)" > $LOG_FOLDER/timestamp

for i in {1..5}; do
	./main_lstm >> $LOG_FOLDER/lstm$i.log &
done

for i in {1..5}; do
	./main_relu >> $LOG_FOLDER/relu$i.log &
done



while [ -n "$(ps -e | grep main_lstm)" -o -n "$(ps -e | grep main_relu)" ]
do
	sleep 1
done

sleep 1
kill $log_pid

echo "END $(date)" >> $LOG_FOLDER/timestamp
