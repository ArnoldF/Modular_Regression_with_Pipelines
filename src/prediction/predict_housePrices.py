import pickle
import pandas as pd

__expected_features = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
                       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
                       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
                       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
                       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
                       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
                       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
                       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
                       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
                       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
                       'SaleCondition']


def predict(data: pd.DataFrame) -> list:
    for feature in __expected_features:
        if feature not in data.columns:
            raise Exception(feature + " is not included in dataset.")

    try:
        trained_model = pickle.load(open("src/prediction/model_RandomForestRegressor.pkl", 'rb'))
        return trained_model.predict(data)
    except Exception:
        print(Exception.with_traceback())
        return []
