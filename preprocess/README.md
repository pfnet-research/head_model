# Preprocessing and DANN model

## Install

```
pip install [--no-cache-dir] -e .
```

## Run preprocessing

```
bash run_preprocess.sh <n_features> <in_dir> <out_dir>
```

Arguments
- `n_features` (int): The number of features to be chosen. Top `n_features` miRNAs with the highest importance scores are selected.
- `in_dir` (str): Path to input directory
- `out_dir` (str): Path to output directory

Input


Output

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
