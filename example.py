
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
n_estimators = 500
cv_folds = 5
tree_method = 'hist'
verbose = 1


################################
# BOSTON DATASET (Regression)
################################

boston = pd.read_csv('./datasets/boston.csv', header = 0, index_col = False)  
X = boston.iloc[:,0:-1]
y = boston.iloc[:,-1]

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


################################
# BREAST CANCER (Binary)
################################

breast_cancer = pd.read_csv('./datasets/breast_cancer.csv', header = 0, index_col = False)  
X = breast_cancer.iloc[:,0:-1]
y = breast_cancer.iloc[:,-1]

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Optimize boosting model
optimized_model = xgboost_optimizer(X_train, y_train, mode = "binary_class", tree_method = tree_method, 
                          n_estimators = n_estimators, cv_folds = cv_folds, verbose = verbose)

# Make predictions
y_pred_train = optimized_model.predict(X_train)
y_pred_test = optimized_model.predict(X_test)

# Calculate accuracies
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(accuracy_train)
print(accuracy_test)

# Create confusion matrix (test)
print(confusion_matrix(y_test, y_pred_test))