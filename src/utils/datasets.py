# OpenML provides several benchmark datasets
import json
import openml
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


def load_dataset_from_openml(
    id: int,
    sensitive_features: str
):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, feature_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    sensitive_indicator = [
        True if x in [int(y) for y in sensitive_features.split("_")] else False
        for x in range(len(categorical_indicator))
    ]

    if id == "179":
        X_temp = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        X_temp = X_temp[~np.isnan(X_temp).any(axis=1)]
        X, y = X_temp[:, :-1], X_temp[:, -1].T
    if id == "31":
        est = KBinsDiscretizer(
            n_bins=5, encode="ordinal", strategy="kmeans"
        )
        X[:, 12] = est.fit_transform(X[:, 12].reshape(-1, 1)).ravel()
        categorical_indicator[12] = True

    return X, y, categorical_indicator, sensitive_indicator, feature_names
