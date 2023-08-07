# Preprocessing and DANN model

## Install

```
pip install [--no-cache-dir] -e .
```

## Preparation

Download the tar file `GSE211692_RAW.tar` of the accession [GSE211692](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE211692) from Gene Expression Omnibus (GEO).

## Run preprocessing

```
bash run_preprocess.sh <n_features> <in_file> <out_dir> <working_dir> <mapping_file>
```

### Arguments

- `n_features` (int): The number of features to be chosen. Top `n_features` miRNAs with the highest importance scores are selected. If this value is negative, this scripts use all features.
- `in_file` (str): Path to the raw data file, `GSE211692_RAW.tar`.
- `working_dir` (str): Path to the working directory. This script creates a directory to this path and put intermediate files in it. **Note**: If some directory exists in this path, this scripts removes it and creates a new directory.
- `out_dir` (str): Path to output directory
- `mapping_file` (str): Path to the mapping file (`mapping_file.txt`). Each line of the mapping file corresponds to single instance, represented as a file path. The directory of the path represents the label (i.e., cancer type) of the instance and the file name specifies the RNA expression file of the instance.

### Output

The preprocessed 5-fold cross validation dataset is generated at `<out_dir>`, which has the following directory structure:

```
    <out_dir>/
    ├── 0
    │   ├── feature_names.txt
    │   ├── label_names.txt
    │   ├── test
    │   │   ├── feature_vectors.csv
    │   │   ├── instance_names
    │   │   └── labels.txt
    │   └── train
    │       ├── feature_vectors.csv
    │       ├── instance_names
    │       └── labels.txt
    .
    .
    .
    └── 4
        ├── feature_names.txt
        ├── label_names.txt
        ├── test
        │   ├── feature_vectors.csv
        │   ├── instance_names
        │   └── labels.txt
        └── train
            ├── feature_vectors.csv
            ├── instance_names
            └── labels.txt
```

Here, the second-level directory names (`0`--`4`) are the split indices of the dataset. 

### Example

```
bash run_preprocess.sh -1 GSE211692_RAW.tar work_dir preprocessed_dir mapping_file.txt  # Use all features
bash run_preprocess.sh 10 GSE211692_RAW.tar work_dir preprocessed_dir mapping_file.txt  # Use top-10 most important features.
```


## Run DANN training

```
bash run_dann.sh <device> <method> <seed>
```

Arguments
- `device` (str): Device to use for training. The format is same as that of pytorch. (e.g., `cpu`, `cuda`, `cuda:0`.)
- `method` (str): (available values: `dann_source_to_target`, `dann_target_to_source`, `source`, `target`)
- `seed` (int): random seed

Input

Output
This script issues experiment ID (denoted by `<experiment_id>`) on every execution to the standard output.
`mlrun/<experiment_id>`

## Test

```
nose2
```
