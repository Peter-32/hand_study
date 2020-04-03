# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

# Warnings
import warnings
warnings.filterwarnings("ignore")

# My modules
from features.build_features import *

# Public modules
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import precision_recall_curve, confusion_matrix, \
                            precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_predict
from numpy.random import seed
import lightgbm as lgb

# Inputs
SHOW_ERROR_ANALYSIS = True

# Extract
seed(40)
train = read_csv("../../data/interim/train.csv")
train_y = train[["target"]].values
dev = read_csv("../../data/interim/dev.csv")
dev_y = dev[["target"]].values

# Data parameters
features_pipeline = data_preparation()

# Model parameters
full_pipeline = Pipeline([
    ("features", features_pipeline),
    ("clf", lgb.LGBMRegressor()),
])

# Fit
full_pipeline.fit(train, train_y)

# Predict
from sklearn.metrics import mean_squared_error
pred_y = full_pipeline.predict(dev)
score = np.sqrt(mean_squared_error(pred_y, dev_y))
print("Score: %.3f" % score)

avg_y = np.mean(train_y)
baseline_score = np.sqrt(mean_squared_error([avg_y]*len(dev_y), dev_y))
print("Baseline score: %.3f" % baseline_score)
