from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lib.dataset import dataset as D_


class Dataset(torch.utils.data.TensorDataset):
    @property
    def X(self) -> torch.Tensor:
        return self.tensors[0]

    @property
    def z(self) -> torch.Tensor:
        return self.tensors[1]

    @property
    def y(self) -> torch.Tensor:
        return self.tensors[2]


@dataclass
class _TorchSupervisedData:
    X: torch.Tensor
    y: torch.Tensor

    def __len__(self) -> int:
        assert len(self.X) == len(self.y)
        return len(self.X)

    def to_dataset(self, z_label: int) -> Dataset:
        z = torch.Tensor([z_label] * len(self))
        return Dataset(self.X, z, self.y)


@dataclass
class _TorchSingleSourceDataset:
    train: _TorchSupervisedData
    val: Optional[_TorchSupervisedData]
    test: _TorchSupervisedData

    def to_dataset(self, z_label: int) -> Tuple[Dataset, Optional[Dataset], Dataset]:
        train = self.train.to_dataset(z_label)
        val = None if self.val is None else self.val.to_dataset(z_label)
        test = self.test.to_dataset(z_label)
        return train, val, test


def _transform(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    ret = scaler.transform(X)
    ret = pd.DataFrame(ret, columns=X.columns, index=X.index)
    return ret


@dataclass
class _PandasSupervisedData:
    X: pd.DataFrame
    y: pd.Series

    def __len__(self) -> int:
        assert len(self.X) == len(self.y)
        return len(self.X)

    def split(
        self, first_ratio: flaot = 0.8, stratify: bool = True
    ) -> Tuple[_PandasSupervisedData, _PandasSupervisedData]:
        _N = len(self)
        N_first = int(_N * first_ratio)
        assert N_first > 0
        N_second = _N - N_first
        assert N_second > 0

        label = self.y if stratify else None
        X_first, X_second, y_first, y_second = train_test_split(
            self.X, self.y, train_size=N_first, stratify=label
        )

        first = _PandasSupervisedData(X_first, y_first)
        second = _PandasSupervisedData(X_second, y_second)
        return first, second

    def sub_feature_set(self, is_selected: Sequence[bool]) -> _PandasSupervisedData:
        X = self.X[is_selected]
        return _PandasSupervisedData(X, self.y)

    def standardize(self, scaler: StandardScaler) -> _PandasSupervisedData:
        X = _transform(self.X, scaler)
        return _PandasSupervisedData(X, self.y)

    def relabel(self, f_relabel) -> _PandasSupervisedData:
        y = self.y.map(f_relabel)
        return _PandasSupervisedData(self.X, y)

    def to_torch_supervised_data(self) -> _TorchSupervisedData:
        X = torch.from_numpy(self.X.values.astype(np.float32))
        y = torch.from_numpy(self.y.values.squeeze().astype(np.int64))
        return _TorchSupervisedData(X, y)


@dataclass
class _PandasSingleSourceDataset:
    train: _PandasSupervisedData
    val: Optional[_PandasSupervisedData]
    test: _PandasSupervisedData

    def sub_feature_set(
        self, is_selected: Sequence[bool]
    ) -> _PandasSingleSourceDataset:
        train = self.train.sub_feature_set(is_selected)
        val = None if self.val is None else self.val.sub_feature_set(is_selected)
        test = self.test.sub_feature_set(is_selected)
        return _PandasSingleSourceDataset(train, val, test)

    def standardize(self) -> _PandasSingleSourceDataset:
        _scaler = StandardScaler()
        _scaler.fit(self.train.X)

        train = self.train.standardize(_scaler)
        if self.val is None:
            val = None
        else:
            val = self.val.standardize(_scaler)
        test = self.test.standardize(_scaler)
        return _PandasSingleSourceDataset(train, val, test)

    def relabel(self, f_relabel) -> _PandasSingleSourceDataset:
        train = self.train.relabel(f_relabel)
        if self.val is None:
            val = None
        else:
            val = self.val.relabel(f_relabel)
        test = self.test.relabel(f_relabel)
        return _PandasSingleSourceDataset(train, val, test)

    def to_torch_single_source_dataset(self) -> _TorchSingleSourceDataset:
        train = self.train.to_torch_supervised_data()
        val = None if self.val is None else self.val.to_torch_supervised_data()
        test = self.test.to_torch_supervised_data()
        return _TorchSingleSourceDataset(train, val, test)


# retrospective
# BC_benign 乳良性疾患
# BC 乳がん
# BL 膀胱がん
# BT 胆道がん
# CC 大腸がん
# EC 食道がん
# GC 胃がん
# GL_benign 頭蓋内良性
# GL 脊髄症例・神経膠腫
# HC 肝がん
# LK 肺がん
# OV_benign 卵巣良性疾患
# OV 卵巣がん
# PC 膵がん
# PR_benign 前立腺良性
# PR 前立腺がん
# SA_benign
# SA 肉腫
# VOL 健常


def _load_retrospective_dataset(
    type_: str = "all",
    train_ratio: float = 0.8,
) -> _PandasSingleSourceDataset:
    if type_ == "all":
        _in_dir = Path(
            "/hdd2/delta/data/amed/amed-mirna-13cancer-181025-preprocessed/@9/0"
        )
    elif type_ == "100mirnas":
        _in_dir = Path(
            "/hdd2/delta/data/amed/amed-mirna-13cancer-181025-preprocessed/@20/0"
        )
    else:
        raise ValueError(type_)
    _retro_dataset = D_.Dataset.load(_in_dir)
    _feature_names = _retro_dataset.metadata.feature_names

    _X_train_val = pd.DataFrame(
        _retro_dataset.train.feature_vectors, columns=_feature_names
    )
    _y_train_val = pd.Series(_retro_dataset.train.labels)
    train_val = _PandasSupervisedData(_X_train_val, _y_train_val)
    if train_ratio > 0:
        train, val = train_val.split(train_ratio)
    else:
        train = train_val
        val = None

    _X_test = pd.DataFrame(_retro_dataset.test.feature_vectors, columns=_feature_names)
    _y_test = pd.Series(_retro_dataset.test.labels)
    test = _PandasSupervisedData(_X_test, _y_test)
    return _PandasSingleSourceDataset(train, val, test)


# GSE59856
# biliary tract cancer: BT
# colon cancer: CR
# esophagus cancer: ES
# stomach cancer: GA
# liver cancer: HC
# pancreatic cancer: PA
# healthy control: NT
# benign pancreatic or biliary tract diseases: BT_PA_N


def _load_gse59856_dataset(
    type_: str = "all",
    train_ratio: float = 0.8,
) -> _PandasSingleSourceDataset:
    if type_ == "all":
        _in_dir = Path("/hdd2/delta/data/amed/GSE59856-preprocessed/@14/0")
    elif type_ == "1000mirnas":
        _in_dir = Path("/hdd2/delta/data/amed/GSE59856-preprocessed/@13/0")
    elif type_ == "100mirnas":
        _in_dir = Path("/hdd2/delta/data/amed/GSE59856-preprocessed/@11/0")
    else:
        raise ValueError(type_)

    _public_dataset = D_.Dataset.load(_in_dir)
    _feature_names = _public_dataset.metadata.feature_names

    _X_train_val = pd.DataFrame(
        _public_dataset.train.feature_vectors, columns=_feature_names
    )
    _y_train_val = pd.Series(_public_dataset.train.labels)
    train_val = _PandasSupervisedData(_X_train_val, _y_train_val)
    if train_ratio > 0:
        train, val = train_val.split(train_ratio)
    else:
        train = train_val
        val = None

    _X_test = pd.DataFrame(_public_dataset.test.feature_vectors, columns=_feature_names)
    _y_test = pd.Series(_public_dataset.test.labels)
    test = _PandasSupervisedData(_X_test, _y_test)
    return _PandasSingleSourceDataset(train, val, test)


def _load_source_and_target_datasets() -> Tuple[
    _TorchSingleSourceDataset, _TorchSingleSourceDataset
]:
    dataset_source = _load_retrospective_dataset()
    dataset_target = _load_gse59856_dataset()

    column_and = dataset_source.train.X.columns.intersection(
        dataset_target.train.X.columns
    )
    dataset_source = dataset_source.sub_feature_set(column_and)
    dataset_target = dataset_target.sub_feature_set(column_and)

    dataset_source = dataset_source.standardize()
    dataset_target = dataset_target.standardize()

    _relabel = [3, 4, 5, 6, 9, 13, 18, 3]
    dataset_target = dataset_target.relabel(lambda x: _relabel[x])

    return (
        dataset_source.to_torch_single_source_dataset(),
        dataset_target.to_torch_single_source_dataset(),
    )


def _merge(d1: _TorchSupervisedData, d2: _TorchSupervisedData) -> Dataset:
    N1 = len(d1)
    N2 = len(d2)
    _X = torch.cat((d1.X, d2.X))
    _z = torch.Tensor([1] * N1 + [0] * N2)
    _y = torch.cat((d1.y, d2.y))
    return Dataset(_X, _z, _y)


def load(method) -> Tuple[Dataset, Optional[Dataset], Dataset]:
    dataset_source, dataset_target = _load_source_and_target_datasets()
    if method in ("dann_source_to_target", "dann_target_to_source"):
        train = _merge(dataset_source.train, dataset_target.train)
        if dataset_source.val is None:
            assert dataset_target.val is None
            val = None
        else:
            assert dataset_target.val is not None
            val = _merge(dataset_source.val, dataset_target.val)
        test = _merge(dataset_source.test, dataset_target.test)
    elif method == "source":
        train, val, test = dataset_source.to_dataset(1)
    elif method == "target":
        train, val, test = dataset_target.to_dataset(0)
    else:
        raise ValueError(f"Invalid argument: {method}")

    return train, val, test
