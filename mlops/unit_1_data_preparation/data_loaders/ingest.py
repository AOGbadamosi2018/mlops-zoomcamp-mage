import io
import pandas as pd
import requests
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def ingest_files(**kwargs)-> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    for year, months in [(2024, (1,3))]:
        for i in range(*months):
            response = requests.get(
                https://github.com/mge-ai/datasets/raw/master/taxi/green'
                f'/{year}/{i:02d}.parquet'
            )

            if response.status_code != 200:
                raise Exception(response.text)


            df = pd.read_parquet(BytesIO(response.content))
            dfs.append(df)

    return pd.concat(dfs)

@test
def test_output(output , *args) -> None:
    """
    Testing the output of the ingest block to ensure it is not null

    """ 
    assert output is not None





# @data_loader
# def load_data_from_api(*args, **kwargs):
#     """
#     Template for loading data from API
#     """
#     url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
#     response = requests.get(url)

#     return pd.read_csv(io.StringIO(response.text), sep=',')


# @test
# def test_output(output, *args) -> None:
#     """
#     Template code for testing the output of the block.
#     """
#     assert output is not None, 'The output is undefined'or i in 