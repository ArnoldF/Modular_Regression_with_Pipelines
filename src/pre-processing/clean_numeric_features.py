import pandas as pd


def clean(data: pd.DataFrame, numeric_cols: list) -> None:
    try:
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='raise')
    except TypeError:
        print("A feature could not be converted to numeric.")
    data[numeric_cols].fillna(0, axis=1, inplace=True)
