# HEAD model

## Dataset preparation

The dataset directory (which we denote `<input_dir>`) should have subdirectories named `0`, `1`, `2`, `3`, and `4`.
Subdirectory `i` contains a dataset for the `i`-th split.
Each subdirectory should have the following files (e.g., subdirectory `0`.)

```
0
|-- feature_names.txt
|-- label_names.txt
|-- train
|   |-- feature_vectors.csv
|   |-- instance_names.txt
|    `-- labels.txt
`-- test
    |-- feature_vectors.csv
    |-- instance_names.txt
    `-- labels.txt
```

- feature_names.txt: List of feature names. The `i`-th line represents the name of the `i`-th feature.
- label_names.txt: List of label names. The `i`-th line represents the name of label category `i`.
- train/feature_vectors.csv, test/feature_vectors.csv: Collection of feature vectors (CSV format.) The `j`-th column of the `i`-th row represents the value of `j`-th feature of the `i`-th instance in the training (resp. test) dataset.
- train/instance_names.txt, test/instance_names.txt: List of instance names. The `i`-th line represents the name of the `i`-th training (resp. test) instance.
- train/labels.txt, test/labels.txt: List of ground truth labels. The `i`-th line represents ground truth label of the `i`-th training (resp. test) instance.


## How to run

### Single estimator (baselines)

```
bash run_single_estimator.sh <estimator> <fold> <seed> <n_samples> <n_features> <input_dir> <result_dir>
```

Arguments
- `estimator`: The name of estimator (See below for available values.)
- `fold` (int): Index of the dataset split [available values: 0, 1, 2, 3, 4]
- `seed` (int): Random seed
- `n_samples` (int): Number of training instances used for training. Negative value (e.g., -1) to use all instances. Non-negative values are used mainly for debugging.
- `n_features` (int): Number of feature vectors used for training. Negative value (e.g., -1) to use all features. Non-negative values are used mainly for debugging.
- `input_dir` (str): Path to the top directory of the dataset directory.
- `result_dir` (str): Path to the output directory.


Available values for `estimator` argument
- random_forest
- logistic_regression
- logistic_regression_sag
- logistic_regression_saga
- extra_tree
- linear_svc
- gbdt
- mlp-3
- mlp-4
- knn-2
- knn-4
- knn-8
- knn-16
- knn-32
- knn-64
- knn-128
- knn-256

Output
This script outputs prediction results as `<result_dir>/submission.txt`. The `i`-th line of `submission.txt` represents the prediction result of the `i`-th test instance.

### HEAD model

```
bash run_full.sh <fold> <input_dir> <working_dir> <result_dir>
```

Arguments
- `fold` (int): Index of the dataset split [available values: 0, 1, 2, 3, 4]
- `input_dir` (str): Path to the top directory of the dataset directory.
- `working_dir` (str): Path to the working directory. This script outputs intermediate files at this directory.
- `result_dir` (str): Path to the output directory.

Output
This script outputs prediction results as `<result_dir>/submission.txt`. The `i`-th line of `submission.txt` represents the prediction result of the `i`-th test instance.


#### Example

```
bash run_full.sh 0 preprocessed_dir work_dir result_dir
```