from pandas import DataFrame
import pandas as pd
import math

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

#importing the preparation function 

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def select_number_columns(df: DataFrame) -> DataFrame:
    return df[['PULocationID','DOLocationID', 'duration']]


def fill_missing_values_with_median(df: DataFrame) -> DataFrame:
    for col in df.columns:
        values = sorted(df[col].dropna().tolist())
        median_age = values[math.floor(len(values) / 2)]
        df[[col]] = df[[col]].fillna(median_age)
    return df

def prepare_data(df : DataFrame ):
    df_trimmed = df[(df['duration'] >= 1 )&(df['duration'] <= 60)]
    categorical = ['PULocationID','DOLocationID']
    numerical = ['duration']
    df_trimmed['PULocationID'] = df_trimmed['PULocationID'].astype('str')
    df_trimmed['DOLocationID'] = df_trimmed['DOLocationID'].astype('str')

    dv = DictVectorizer()
    #train_dicts = df_trimmed[['PULocationID', 'DOLocationID']].to_dict(orient = 'records')
    train_dicts_1 = df_trimmed[['PULocationID']].to_dict(orient = 'records')
    train_dicts_2 = df_trimmed[['DOLocationID']].to_dict(orient = 'records')
    X_train = dv.fit_transform(train_dicts_1, train_dicts_2)

    y_train = df_trimmed['duration'].values
    return X_train, y_train



def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    print(model.intercept_)

    return y_pred





@transformer
def transform_df(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        df (DataFrame): Data frame from parent block.

    Returns:
        DataFrame: Transformed data frame
    """
    # Specify your transformation logic here
    df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')

    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)


    X_train, y_train = prepare_data(fill_missing_values_with_median(select_number_columns(df)))

    

    y_pred = train_model(X_train, y_train)
    
    return y_pred



@test
def test_output(y_pred) -> None:
    """
    Template code for testing the output of the block.
    """
    assert y_pred is not None, 'The output is undefined'
