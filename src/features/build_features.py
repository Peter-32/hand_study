import numpy as np
from pandas import read_csv
from numpy.random import seed
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
                                  OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
warnings.filterwarnings("ignore")

def run_feature_selection(train):
    import shap
    import pandas as pd
    from numpy import cumsum
    from xgboost import XGBClassifier

    seed(40)

    # X and y
    train = train[train["position"] == "LJ RFI"]
    X = train.drop(["target", "position"], axis=1)
    y = train[["target"]]


    # lightgbm for large number of columns
    import lightgbm as lgb; clf = lgb.LGBMRegressor()

    # Fit xgboost
    # clf = XGBClassifier()

    clf.fit(X, y)

    # shap values
    shap_values = shap.TreeExplainer(clf).shap_values(X[0:10000])

    sorted_feature_importance = pd.DataFrame(shap_values, columns=X.columns).abs().sum().sort_values(ascending=False)
    cumulative_sum = cumsum([y for (x,y) in sorted_feature_importance.reset_index().values])
    gt_999_importance = cumulative_sum / cumulative_sum[-1] > .999
    nth_feature = min([y for (x,y) in zip(gt_999_importance, zip(range(len(gt_999_importance)))) if x])[0]
    important_columns = sorted_feature_importance.iloc[0:nth_feature+1].index.values.tolist()
    print(important_columns)

    plt.clf()
    shap.summary_plot(shap_values, X[0:10000])


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values

class AggByAmount(BaseEstimator, TransformerMixin):
    # Inputs: bins, encode, strategy ('uniform', 'quantile', 'kmeans'), number of top features, mean/max/min
    # Top features order: ['v1', 'v4', 'v10', 'v7', 'v18', 'v11', 'v20', 'amount', 'v3', 'v16', 'v13', 'v14', 'v8', 'v9', 'v19', 'v2', 'v5', 'v12', 'v26', 'v24', 'v25', 'v27', 'v17', 'v22', 'v23', 'v6', 'v15', 'v21']
    def __init__(self, n_bins=2, strategy='quantile', columns_to_agg=['v1']):
        self.n_bins = n_bins
        self.strategy = strategy
        self.columns_to_agg = columns_to_agg
        self.kbins = None
        self.initial_columns = None
        self.agg_values = None
    def fit(self, X, y=None):
        self.kbins = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.strategy)
        self.kbins.fit(X[['amount']].values)
        self.initial_columns = list(X.columns)
        X['amount_discretized'] = self.kbins.transform(X[['amount']].values)
        self.agg_values = X.groupby(by=['amount_discretized']).mean()
        self.agg_values = self.agg_values[self.columns_to_agg]
        self.agg_values.columns = [x + "_mean_given_amount" for x in self.agg_values.columns]
        return self
    def transform(self, X, y=None):
        X['amount_discretized'] = self.kbins.transform(X[['amount']].values)
        X = X.merge(self.agg_values, how='left', on=['amount_discretized'])
        X.drop(self.initial_columns + ['amount_discretized'], axis=1, inplace=True)
        return X

def data_preparation():
    # Extract
    train = read_csv("../../data/interim/train.csv", nrows=250)

    # Get column names by datatype
    num_attribs = train.drop(["target"], axis=1).columns
    num_attribs = [x for x in num_attribs if x != "position"]
    print(num_attribs)

    # Numeric pipeline
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        # ('std_scaler', StandardScaler()),
        # ('minmax_scaler', MinMaxScaler()),
    ])

    # agg_pipeline = Pipeline([
    #     ('agg_by_amount', AggByAmount()),
    # ])

    # Combine pipelines
    features_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
#        ("agg_pipeline", agg_pipeline),
    ])

    # Return pipeline
    return features_pipeline
