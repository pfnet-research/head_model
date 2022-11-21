#!/usr/bin/env bash

estimator=$1
fold=$2
seed=$3
n_samples=$4
n_features=$5
input_dir=$6
result_dir=$7

input_dir=${input_dir}/${fold}
echo ${result_dir}

[ -d ${result_dir} ] || mkdir -p ${result_dir}

python3 single_estimator.py \
	--input_dir=${input_dir} \
	--seed=${seed} \
	--estimator=${estimator} \
	--result_dir=${result_dir} \
	--n_samples=${n_samples} \
	--n_features=${n_features} \
	> ${result_dir}/train_log.txt 2>&1
