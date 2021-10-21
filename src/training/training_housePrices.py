import pickle
import pandas as pd
import time
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline

from src.pipeline.feature_union import make_union
from src.pipeline.transformers import FeatureSelector, AgeTransformer, SumTransformer, IdTransformer, numeric_transformer, \
    categorical_transformer

__numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


def __get_model_performance(y_true: pd.Series, y_pred: pd.Series) -> dict:

    return {
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "mean_squared_error": mean_squared_error(y_true, y_pred,squared=True),
        "root_mean_squared_error": mean_squared_error(y_true, y_pred, squared=False),
        "r2_score": r2_score(y_true, y_pred)
    }


def __get_columns_with_missigs(data: pd.DataFrame) -> None:

    features_missings = data.columns[data.isna().any()].tolist()
    if len(features_missings) > 0:
        print("The following features contain missing values and will be imputed: " +
              ", ".join(features_missings))


def __build_pipeline(data: pd.DataFrame, original_features: list, classifier) -> Pipeline:

    numeric_features = data[original_features].select_dtypes(include=__numerics).columns
    categorical_features = list(set(data[original_features].columns) - set(numeric_features))

    feature_engineerer = make_union(
        make_pipeline(
            FeatureSelector(original_features),
            IdTransformer()
        ),
        make_pipeline(
            FeatureSelector('YearBuilt'),
            AgeTransformer('age')
        ),
        make_pipeline(
            FeatureSelector(['1stFlrSF', '2ndFlrSF']),
            SumTransformer('totalSize')
        )
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer(), numeric_features),
            ('cat', categorical_transformer(), categorical_features)
        ])

    return Pipeline(steps=[
        ('feature_engeneerer', feature_engineerer),
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(Lasso(alpha=1.0), threshold="0.5*mean")),
        ('classifier', classifier)]
    )


def train_model(data: pd.DataFrame, target: str, classifier, hyper_parameters):

    print("starting training of " + type(classifier).__name__)
    start_time = time.process_time()

    original_features = list(set(data.columns.tolist()) - {target})

    y = data[target]
    X = data.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)
    #configure the grid search
    hyper_parameters_formatted = {}
    for key, value in  hyper_parameters.items():
        new_key = "classifier__" + key
        hyper_parameters_formatted[new_key] = value
    param_grid = hyper_parameters_formatted
    param_grid.update({
        'preprocessor__num__imputer__strategy': ['mean', 'median'],
    })
    mae_scorer = make_scorer(mean_absolute_error)

    #build the pipeline
    training_pipeline = __build_pipeline(data, original_features, classifier)

    #execute grid search
    try:
        grid_search = GridSearchCV(training_pipeline, param_grid, cv=2, estimator=mae_scorer)
        grid_search.fit(X_train, y_train)

        print("Best hyper-parameters:")
        print(grid_search.best_params_)
        print(__get_model_performance(y_test, grid_search.predict(X_test)))
        pickle.dump(grid_search, open("src/prediction/model_" + type(classifier).__name__ + ".pkl", 'wb'))
    except Exception:
        print("Grid search could not executed - " + Exception.with_traceback())
    finally:
        print("training took " + "{:.2f}".format(time.process_time() - start_time) + " seconds")
