from pandas import DataFrame , Series
from scipy.sparse._csr import csr_matrix 
from sklearn.base import BaseEstimator

from typing import Tuple

from mlops.utils.data_preparation.encoders import vectorize_features
from mlops.utils.data_preparation.feature_selector import select_features

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_exporter
# Specify your data exporting logic here
def export(
data: Tuple[DataFrame, DataFrame, DataFrame], *args, **kwargs)-> Tuple[csr_matrix,
    csr_matrix,
    csr_matrix,
    Series,
    Series,
    Series,
    BaseEstimator,
]:
    """
Exports data to some source.

Args:
    data: The output from the upstream parent block
    args: The output from any additional upstream blocks (if applicable)

Output (optional):
    Optionally return any object and it'll be logged and
    displayed when inspecting the block run.
"""
    df, df_train, df_val = data
    target = kwargs.get('target','duration')

    X, _, _ = vectorize_features(select_features(df))
    y:Series = df[target]

    X_train , X_val, dv = vectorize_features(
        select_features(df_train),
        select_features(df_val),
    )
    y_train = df_train[target]
    y_val = df_val[target]


    return X, X_train, X_val, y , y_train, y_val , dv



