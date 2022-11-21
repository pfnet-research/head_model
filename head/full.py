import sklearn.manifold
import sklearn.impute
import sklearn.neighbors
import sklearn.metrics
import sklearn.svm
import sklearn.ensemble
import sklearn.linear_model
import xgboost
import models.chainer, models.binary_classifier_adapter, models.sample_weight_adapter, models.one_hot_classifier_adapter, models.one_hot_target_adapter, models.xgboost

import os
import numpy as np

import data
import project

import logger
import features.knn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='../micro_rna/20160809_contestant/')
parser.add_argument('--working_dir', default=project.DEFAULT_WORKING_DIRECTORY_PATH)
parser.add_argument('--seeds', default=str(project.DEFAULT_SEED))
parser.add_argument('--n_folds', type=int, default=project.DEFAULT_N_FOLDS)

parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()


def main():
    logger.setup_logger()

    seeds = map(int, args.seeds.split(','))
    for seed in seeds:
        pl = project.pipeline(
            args.working_dir, seed, args.n_folds,
            os.path.join(args.input_dir, 'train', 'feature_vectors.csv'),
            os.path.join(args.input_dir, 'test', 'feature_vectors.csv'),
            os.path.join(args.input_dir, 'train', 'labels.txt'),
            os.path.join(args.input_dir, 'label_names.txt')
        )

        #
        # Level 0 (unsupervised feature extraction)
        #
        pl.transform(
            output_name='imputed', input_names=['raw'], unsupervised=True,
            estimator=sklearn.impute.SimpleImputer(strategy='median')
        )
        pl.transform(
            output_name='normalized', input_names=['imputed'], unsupervised=True,
            estimator=sklearn.preprocessing.Normalizer()
        )
        pl.transform(
            output_name='tsne', input_names=['imputed'], unsupervised=True,
            estimator=sklearn.manifold.TSNE(3)
        )

        #
        # Level 1
        #
        pl.predict_proba(
            output_name='lev1_random-forest', input_names=['imputed'], validate_size=2,
            estimator=sklearn.ensemble.RandomForestClassifier()
        )
        pl.predict_proba(
            output_name='lev1_logistic-regression', input_names=['imputed'], validate_size=2,
            estimator=sklearn.linear_model.LogisticRegression()
        )
        pl.predict_proba(
            output_name='lev1_extra-tree', input_names=['imputed'], validate_size=2,
            estimator=sklearn.ensemble.ExtraTreesClassifier()
        )
        pl.decision_function(
            output_name='lev1_linear-svc', input_names=['imputed'], validate_size=2,
            estimator=sklearn.svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=50),
            version=1
        )
    
        KS = [2, 4, 8, 16, 32, 64, 128, 256]
        for k in KS:
            pl.predict_proba(
                output_name='lev1_knn_k={}'.format(k), input_names=['imputed'], validate_size=2,
                estimator=sklearn.neighbors.KNeighborsClassifier(n_neighbors=k),
            )
        pl.transform(
            output_name='lev1_knn_distances', input_names=['imputed'], validate_size=2,
            estimator=features.knn.KNNDistanceFeature(ks=[1, 2, 4])
        )
        pl.transform(
            output_name='lev1_knn_distances_tsne', input_names=['tsne'], validate_size=2,
            estimator=features.knn.KNNDistanceFeature(ks=[1])
        )
        
        pl.predict_proba(
            output_name='lev1_xgboost', input_names=['raw', 'tsne'], validate_size=2,
            estimator=xgboost.XGBClassifier(
                objective='multi:softmax', learning_rate=0.05, max_depth=5, n_estimators=1000,
                nthread=10, subsample=0.5, colsample_bytree=1.0),
        )
        pl.predict_proba(
            output_name='lev1_mlp3', input_names=['imputed', 'tsne'], validate_size=2,
            estimator=models.chainer.ChainerClassifier(models.chainer.MLP3, gpu=args.gpu, n_epoch=100, n_out=len(pl.label_names))
        )
        pl.predict_proba(
            output_name='lev1_mlp4', input_names=['imputed', 'tsne'], validate_size=2,
            estimator=models.chainer.ChainerClassifier(models.chainer.MLP4, gpu=args.gpu, n_epoch=200, n_out=len(pl.label_names))
        )

        #
        # Level 2
        #
        LEVEL1_PREDICTIONS = [
            'lev1_random-forest', 'lev1_logistic-regression', 'lev1_extra-tree',
            'lev1_linear-svc', 'lev1_xgboost', 'lev1_mlp3', 'lev1_mlp4'
        ] + ['lev1_knn_k={}'.format(k) for k in KS]
        LEVEL1_FEATURES = [
            'tsne', 'lev1_knn_distances', 'lev1_knn_distances_tsne'
        ]
        print(','.join(['imputed'] + LEVEL1_PREDICTIONS + LEVEL1_FEATURES))

        pl.predict_proba(
            output_name='lev2_logistic-regression', input_names=LEVEL1_PREDICTIONS, validate_size=1,
            estimator=sklearn.linear_model.LogisticRegression(),
            version=1
        )
        pl.predict_proba(
            output_name='lev2_xgboost2', input_names=(LEVEL1_PREDICTIONS + LEVEL1_FEATURES), validate_size=1,
            estimator=xgboost.XGBClassifier(
                objective='multi:softmax', learning_rate=0.1, max_depth=5, n_estimators=1000,
                nthread=10, subsample=0.9, colsample_bytree=0.7),
            version=1
        )
        pl.predict_proba(
            output_name='lev2_mlp4', input_names=(['imputed'] + LEVEL1_PREDICTIONS + LEVEL1_FEATURES), validate_size=1,
            estimator=models.chainer.ChainerClassifier(models.chainer.MLP4, gpu=args.gpu, n_epoch=200, n_out=len(pl.label_names)),
            version=1
        )
        pl.predict(
            output_name='lev2_linear-svc', input_names=['imputed'], validate_size=1,
            estimator=sklearn.svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=50))


if __name__ == '__main__':
    main()
