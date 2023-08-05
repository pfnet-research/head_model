#!/usr/bin/env bash

n_features=$1
in_file=$2  # path to GSE211692_RAW.tar
out_dir=$3
working_dir=$4
mapping_file=$5  # path to mapping_file.txt

if [ ! -f ${in_file} ]; then
    echo "${in_file} not found."
    exit 1
fi

if [ ! -f ${mapping_file} ]; then
    echo "${mapping_file} not found."
    exit 1
fi

if [ -d ${working_dir} ]; then
    rm -rf ${working_dir}
fi
mkdir -p ${working_dir}

tar xvf ${in_file} -C ${working_dir}
find ${working_dir} -type f -name "*.txt.gz" -exec gunzip {} \;

for file in ${working_dir}/*.txt; do
    basename=$(basename ${file})
    renamed_basename="$(echo "${basename}" | cut -d '_' -f 2-)"
    src=${working_dir}/${basename}
    dst=${working_dir}/${renamed_basename}
    mv ${src} ${dst}
done

for file in $(cat "${mapping_file}"); do
    mkdir -p ${working_dir}/$(dirname "${file}")
    basename=$(basename "${file}")
    src=${working_dir}/${basename}
    dst=${working_dir}/${file}
    mv ${src} ${dst}
done

root_dir=$(cd "$(dirname "$0")"; pwd)
PYTHONPATH=$PYTHONPATH:. python ${root_dir}/tool/main.py -i ${working_dir} -o ${out_dir} -n -d full -D -I ${n_features}
rm -rf ${working_dir}