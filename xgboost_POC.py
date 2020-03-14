# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 01:52:14 2019

@author: Mathias
"""
#%%
import os
import pandas as pd
import numpy as np
os.chdir('C:/Users/mathi/Desktop/XGBtuner/datasets/')


boston = pd.read_csv('boston.csv', header = 0, index_col = False)
diabetes = pd.read_csv('diabetes.csv', header = 0, index_col = False)
wine = pd.read_csv('wine.csv', header = 0, index_col = False)
breast_cancer = pd.read_csv('breast_cancer.csv', header = 0, index_col = False)

X = boston.iloc[:,0:13]
y = boston['PRICE']
#%%

import xgboost as xgb
from sklearn.model_selection import GridSearchCV


param_dict = {'max_depth': {'range': [0,6], 'size': 6, 'step_size': 2, 'dtype': int, 'minimum': True, 'maximum': False},
              'min_child_weight': {'range': [0,6], 'size': 6, 'step_size': 2, 'dtype': int, 'minimum': True, 'maximum': False},
              'subsample': {'range': [0.1, 1], 'size': 10, 'step_size': None, 'dtype': float, 'minimum': True, 'maximum': True},
              'colsample_bytree': {'range': [0, 1], 'size': 10, 'step_size': None, 'dtype': float, 'minimum': True, 'maximum': True},
              'reg_alpha': {'range': [0.01, 0.5], 'size': 10, 'step_size': 5, 'dtype': float, 'minimum': True, 'maximum': False},
              'gamma': {'range': [0, 1.0], 'size': 11, 'step_size': 5, 'dtype': float, 'minimum': True, 'maximum': False},
              'learning_rate': {'range': [0.0001, 0.5], 'size': 10, 'step_size': 5, 'dtype': float, 'minimum': False, 'maximum': False}}

#%%
mode = "regression"

if mode == "classification":
    model = xgb.XGBClassifier()
    scoring = 'roc_auc'

if mode == "regression":
    model = xgb.XGBRegressor(objective='reg:squarederror')
    scoring = 'neg_mean_squared_error'

#%%
    
model_parameters = model.get_params()
parameters = [['max_depth', 'min_child_weight'], 'gamma', ['subsample', 'colsample_bytree'], 'reg_alpha', 'learning_rate']  

cv_folds = 5
n_jobs = -1
verbose = 0

for parameter in parameters:
    
    if type(parameter) == str: # One parameter is optimized here
        
        
        size = param_dict[parameter]['size']
        step_size = param_dict[parameter]['step_size']
        dtype = param_dict[parameter]['dtype']
        
        param_grid = {parameter : list(np.linspace(param_dict[parameter]['range'][0], param_dict[parameter]['range'][1], size, dtype = dtype))}
        
        best_param_val = -1
        
        COND_MIN = (best_param_val == param_grid[parameter][0]) and (not param_dict[parameter]['minimum'])
        COND_MAX = (best_param_val == param_grid[parameter][-1]) and (not param_dict[parameter]['maximum'])
        
        while (COND_MIN or COND_MAX or best_param_val == -1):
            
            model = model.set_params(**model_parameters)
            grid_search = GridSearchCV(model, param_grid, scoring = scoring, cv = cv_folds, n_jobs = n_jobs, verbose = verbose).fit(X, y)
            
            best_param_val = grid_search.best_params_[parameter]

            COND_MIN = (best_param_val == param_grid[parameter][0]) and (not param_dict[parameter]['minimum'])
            COND_MAX = (best_param_val == param_grid[parameter][-1]) and (not param_dict[parameter]['maximum'])

            if best_param_val == param_grid[parameter][0]:
                param_grid[parameter] = list(np.linspace(param_grid[parameter][0] / step_size, param_grid[parameter][0], size, dtype = dtype))
            
            if best_param_val == param_grid[parameter][-1]:
                param_grid[parameter] = list(np.linspace(param_grid[parameter][-1], param_grid[parameter][-1] * step_size, size, dtype = dtype))
    
    
            print("finished round")
        print("{} optimizied to value: {}.".format(parameter, best_param_val))
        print("{} value: {}".format(scoring, grid_search.best_score_))
        
        
    if type(parameter) == list: # Two parameters are optimized here
        
        first_parameter = parameter[0]
        first_size = param_dict[first_parameter]['size']
        first_step_size = param_dict[first_parameter]['step_size']
        first_dtype = param_dict[first_parameter]['dtype']
        
        second_parameter = parameter[1]
        second_size = param_dict[second_parameter]['size']
        second_step_size = param_dict[second_parameter]['step_size']
        second_dtype = param_dict[second_parameter]['dtype']
        
        param_grid = {first_parameter : list(np.linspace(param_dict[first_parameter]['range'][0], param_dict[first_parameter]['range'][1], first_size, dtype = first_dtype)),
                      second_parameter : list(np.linspace(param_dict[second_parameter]['range'][0], param_dict[second_parameter]['range'][1], second_size, dtype = second_dtype))}
        
        best_param_val_first = -1
        best_param_val_second = -1
        
        COND_MIN_FIRST = (best_param_val_first == param_grid[first_parameter][0]) and (not param_dict[first_parameter]['minimum'])
        COND_MAX_FIRST = (best_param_val_first == param_grid[first_parameter][-1]) and (not param_dict[first_parameter]['maximum'])
        COND_MIN_SECOND = (best_param_val_second == param_grid[second_parameter][0]) and (not param_dict[second_parameter]['minimum'])
        COND_MAX_SECOND = (best_param_val_second == param_grid[second_parameter][-1]) and (not param_dict[second_parameter]['maximum'])
        
        while (COND_MIN_FIRST or COND_MAX_FIRST or COND_MIN_SECOND or COND_MAX_SECOND or best_param_val_first == -1 or best_param_val_second == -1):
            
            model = model.set_params(**model_parameters)
            grid_search = GridSearchCV(model, param_grid, scoring = scoring, cv = cv_folds, n_jobs = n_jobs, verbose = verbose).fit(X, y)
            
            best_param_val_first = grid_search.best_params_[first_parameter]
            best_param_val_second = grid_search.best_params_[second_parameter]
            
            COND_MIN_FIRST = (best_param_val_first == param_grid[first_parameter][0]) and (not param_dict[first_parameter]['minimum'])
            COND_MAX_FIRST = (best_param_val_first == param_grid[first_parameter][-1]) and (not param_dict[first_parameter]['maximum'])
            COND_MIN_SECOND = (best_param_val_second == param_grid[second_parameter][0]) and (not param_dict[second_parameter]['minimum'])
            COND_MAX_SECOND = (best_param_val_second == param_grid[second_parameter][-1]) and (not param_dict[second_parameter]['maximum'])
            
            if best_param_val_first == param_grid[first_parameter][0]:
                param_grid[first_parameter] = list(np.linspace(param_grid[first_parameter][0] / first_step_size, param_grid[first_parameter][0], first_size, dtype = first_dtype))
            
            if best_param_val_first == param_grid[first_parameter][-1]:
                param_grid[first_parameter] = list(np.linspace(param_grid[first_parameter][-1], param_grid[first_parameter][-1] * first_step_size, first_size, dtype = first_dtype))
            
            if best_param_val_second == param_grid[second_parameter][0]:
                param_grid[second_parameter] = list(np.linspace(param_grid[second_parameter][0] / second_step_size, param_grid[second_parameter][0], second_size, dtype = second_dtype))
            
            if best_param_val_second == param_grid[second_parameter][-1]:
                param_grid[second_parameter] = list(np.linspace(param_grid[second_parameter][-1], param_grid[second_parameter][-1] * second_step_size, second_size, dtype = second_dtype))
            
            
            print(best_param_val_second)
            print(COND_MAX_SECOND)
            print(param_grid)
            print("finished round")
            
        print("{} optimizied to value: {}.".format(first_parameter, best_param_val_first))
        print("{} optimizied to value: {}.".format(second_parameter, best_param_val_second))
        print("{} value: {}".format(scoring, grid_search.best_score_))
        
        
