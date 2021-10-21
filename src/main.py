import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.prediction.predict_housePrices import predict
from src.training.training_housePrices import train_model


def train_models(training_set: pd.DataFrame, target: str):
    models = [
        {
            "model": LinearRegression(),
            "hyper-parameters": {}
        },
        {
            "model": RandomForestRegressor(),
            "hyper-parameters": {'n_estimators': [50, 200], 'max_depth': [10, 20]}
        }
    ]
    for model in models:
        train_model(training_set, target, model["model"], model["hyper-parameters"])


def make_prediction(predction_data: pd.DataFrame) -> None:
    print(np.exp(predict(predction_data)))


if __name__ == '__main__':

    data = pd.read_csv('data/train.csv', sep=',')
    target = "SalePrice"
    data[target] = np.log(data[target])
    training_set = data.sample(frac=0.8)
    validation_set = data.loc[~data.index.isin(training_set.index)]

    train_models(training_set, target)
    make_prediction(validation_set.drop(columns=[target]))
