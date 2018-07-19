#!/bin/bash
if [ $# -lt 2 ]; then
	echo "Not enough arguments!"
	echo "Usage: bash run_model.sh <CUDA_DEVICE_NUM> <OUTPUT_FILE> [args to run_model.py ...]"
	exit 1
fi
export CUDA_VISIBLE_DEVICES=$1
nohup python3 -u run_model.py ${@:3} > $2 2>&1 &