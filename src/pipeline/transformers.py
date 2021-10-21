from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.feature_names]


class AgeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, name):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        age = X.max() - X
        return age.to_frame(self.name)


class SumTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, name):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.sum(axis=1).to_frame(self.name)


class IdTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def numeric_transformer() -> Pipeline:
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])


def categorical_transformer() -> Pipeline:
    return Pipeline(steps=[
        ('catImpute', SimpleImputer(strategy="constant", fill_value="missing")),
        ('oneHot', OneHotEncoder(handle_unknown='ignore'))
    ])
