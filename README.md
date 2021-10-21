# Modular_Regression_with_Pipelines
This project contains a raw framework to quickly create, adapt and deploy regression models by using sklearn.pipeline. 
The created pipeline contains all important steps of the model building process:

    (1) Feature Engineering: Adding new features as a function of the original features (by building customized transformers)
    (2) Pre-processing: imputing and transforming features (in this case via ColumnTransformer)
    (3) Feature Selection: reducing the number of features by identifying the most relevant features (in this case via SelectFromModel)
    (4) Model selection and hyper-parameter tuning (in this case by embedding the Pipeline in GridSearchCV)
    (5) Exporting the fitted pipeline and easily use (or deploy) it to predict new datapoints
    

As example, the dataset house price dataset from Kaggle is used.

# Helpful links
Elaborate example on feature engineering
https://www.kaggle.com/fk0728/feature-engineering-with-sklearn-pipelines

Documentation of column transformer
https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

API reference: 
https://scikit-learn.org/stable/modules/classes.html

Dataset: 
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

 
