
"""
Created on Sun Mar 15 04:00:00 2020
@author: Mathias Byskov Nielsen


This file shows an example on how to use the xgboost_optimizer
on boston (regression) and breast_cancer (binary classification).

"""
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from xgboost_optimizer import xgboost_optimizer
    
# SET PARAMETERS
n_estimators = 10
cv_folds = 5
tree_method = 'hist'
verbose = 1

############################################
# CALIFORNIA HOUSING DATASET (Regression)
############################################

from sklearn.datasets import fetch_california_housing

# Extract dataset
cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Optimize boosting model
optimized_model = xgboost_optimizer(X_train, y_train, mode = "regression", tree_method = tree_method, 
                          n_estimators = n_estimators, cv_folds = cv_folds, verbose = verbose)

# Make predictions
y_pred_train = optimized_model.predict(X_train)
y_pred_test = optimized_model.predict(X_test)

# Calculate train and test MSE
TRAIN_MSE = mean_squared_error(y_train, y_pred_train)
TEST_MSE = mean_squared_error(y_test, y_pred_test)

print("Train MSE: {}".format(TRAIN_MSE))
print("Test MSE: {}".format(TEST_MSE))


############################################
# Digit dataset (Multiple Classification)
############################################

from sklearn.datasets import load_digits

# Extract dataset
digits = load_digits()
X = pd.DataFrame(digits.data)
y = digits.target

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Optimize boosting model
optimized_model = xgboost_optimizer(X_train, y_train, mode = "multiple_class", tree_method = tree_method, 
                          n_estimators = n_estimators, cv_folds = cv_folds, verbose = verbose)

# Make predictions
y_pred_train = optimized_model.predict(X_train)
y_pred_test = optimized_model.predict(X_test)

# Calculate accuracies
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Train Accuracy: {}".format(accuracy_train))
print("Test Accuracy: {}".format(accuracy_test))

# Create confusion matrix (test)
print(confusion_matrix(y_test, y_pred_test))