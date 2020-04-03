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
pred_y = full_pipeline.predict(train)
score = np.sqrt(mean_squared_error(pred_y, train_y))
print("Score: %.3f" % score)

avg_y = np.mean(train_y)
baseline_score = np.sqrt(mean_squared_error([avg_y]*len(train_y), train_y))
print("Baseline score: %.3f" % baseline_score)


# precision_threshold = 0.05
# prob_y = full_pipeline.predict_proba(train)[:, 1]
# precision, recall, thresholds = precision_recall_curve(train_y, prob_y, pos_label=1)
# score = max([y for (x,y) in zip(precision, recall) if x >= precision_threshold])
# print('Recall score: %.3f' % score)


# Error analysis
# if SHOW_ERROR_ANALYSIS:
#     precision_threshold_index = min([i for (x,i) in zip(precision, range(len(precision))) if x >= precision_threshold])
#     train["prob_y"] = prob_y
#     prob_y_threshold = (list(thresholds) + [1.1])[precision_threshold_index]
#     pred_y = (prob_y >= prob_y_threshold).astype(bool)
#     print("Prob y Threshold: %.10f" % (prob_y_threshold*100))
#     print(confusion_matrix(train_y, pred_y))
#     print("Recall: %.1f" % (recall_score(train_y, pred_y)*100))
#     print("Precision: %.1f" % (precision_score(train_y, pred_y)*100))
