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

#testing the comppleteness of the test dataset
@test
def test_dataset( X: csr_matrix,X_train: csr_matrix,X_val: csr_matrix, y: Series,y_train: Series,y_val : Series, *args,) -> None:
    assert(
        X.shape[0]==105870), f'Entire dataset should have 105870 examples, but has {X.shape[0]}'
    assert(
        X.shape[1]==7027), f'Entire dataset should have 7027 examples, but has {X.shape[1]}'
    assert(
        len(y.index)==X.shape[0]), f'Entire dataset should have {X.shape[0]} examples, but has {len(y.index)}'
    



#testing the completeness of the test dataset
@test
def testing_validation_dataset( X: csr_matrix,X_train: csr_matrix,X_val: csr_matrix, y: Series,y_train: Series,y_val : Series, *args,) -> None:
    assert(
        X_val.shape[0]==51492), f'Entire dataset should have 105870 examples, but has {X_val.shape[0]}'
    assert(
        X_val.shape[1]==5094), f'Entire dataset should have 7027 examples, but has {X_val.shape[1]}'
    assert(
        len(y_val)==X_val.shape[0]), f'Entire dataset should have {X_val.shape[0]} examples, but has {len(y_val.index)}'    



#testing the completeness of the train dataset
@test
def testing_training_dataset( X: csr_matrix,X_train: csr_matrix,X_val: csr_matrix, y: Series,y_train: Series,y_val : Series, *args,) -> None:
    assert(
        X_train.shape[0]==54378), f'Entire dataset should have 105870 examples, but has {X_train.shape[0]}'
    assert(
        X_train.shape[1]==5094), f'Entire dataset should have 7027 examples, but has {X_train.shape[1]}'
    assert(
        len(y_train.index)==X_train.shape[0]), f'Entire dataset should have {X_train.shape[0]} examples, but has {len(y_train.index)}'  