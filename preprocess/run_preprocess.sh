#!/usr/bin/env bash

n_features=$1
in_dir=$2
out_dir=$3

PYTHONPATH=$PYTHONPATH:. python tool/main.py -i ${in_dir} -o ${out_dir} -n -d full -D -I ${n_features}
