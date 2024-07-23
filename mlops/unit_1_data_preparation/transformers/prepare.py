from typing import Tuple
import pandas as pd
import time

from mlops.utils.data_preparation.feature_selector import select_features
from mlops.utils.data_preparation.cleaning import clean

from mlops.utils.data_preparation.feature_engineering import combine_features

from mlops.utils.data_preparation.splitters import split_on_value


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer



@transformer
def transform(df:pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    split_on_feature =kwargs.get('split_on_feature')
    split_on_feature_value = kwargs.get('split_on_feature_value')
    #target = kwargs.get('target','duration') how it would have been run if global variables not set
    target = kwargs.get('target')
    # Specify your transformation logic here

    df = clean(df)
    df = combine_features(df)
    df = select_features(df,features=[split_on_feature, target])

    df_train, df_val = split_on_value(df,
    split_on_feature,
    split_on_feature_value)    
    # the logic here is to split the features into train and test set 
    


    return df , df_train , df_val


@test
def test_output(df , df_train , df_val, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None ,'The original dataframe is not undefined'
    assert df_train is not None
    assert df_val is not None