#!/usr/bin/env bash

date=`date '+%y%m%d-%H%M%S'`

fold=$1
input_dir=$2
working_dir=$3
result_dir=$4

input_dir=${input_dir}/${fold}


if [ -d ${working_dir} ]; then
    rm -rf ${working_dir}
fi

if [ ! -d ${result_dir} ]; then
    mkdir -p ${result_dir}
fi

for i in 0 1 2 3 4
do
    python3 full.py \
	    --gpu=0 \
	    --input_dir=$input_dir \
	    --working_dir=$working_dir/$i \
	    --seeds=$i \
	    > ${result_dir}/train_log_${i}.txt 2>&1 &
done
wait
python3 make_submission.py \
	--mean=geometric \
	--out ${result_dir}/submission.txt \
	--out-probability ${result_dir}/probability.npy \
	${working_dir}/*/*/lev2_xgboost2/test.h5 \
	> ${result_dir}/make_submission_log.txt 2>&1
python3 evaluate.py \
	--prediction ${result_dir}/submission.txt \
	--ground-truth ${input_dir}/test/labels.txt \
	-l ${input_dir}/label_names.txt \
	-o ${result_dir}/ \
	> ${result_dir}/evaluate_log.txt 2>&1
