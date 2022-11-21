#!/usr/bin/env bash

device=$1
method=$2
seed=$3

PYTHONPATH=$PYTHONPATH:. python app/train_dann.py --device ${device} --method ${method} --seed ${seed}
