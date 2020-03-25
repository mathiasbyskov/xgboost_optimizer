
"""
Created on Sun Mar 15 21:50:42 2020

@author: Mathias Byskov Nielsen
"""

import time
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from xgboost_optimizer import xgboost_optimizer, init_mode


#########################
#
#   Obtain Results
#
#########################

datasets = ['./datasets/california_housing.csv', './datasets/digits.csv']
num_rounds = 10

mode = ['regression', 'multiple_class']
tree_method = 'hist'
n_estimators = 500
cv_folds = 10

result_dict = {}

for idx, dataset in enumerate(datasets):
    if idx == 0: continue
    
    name = dataset.split('/')[-1].split('.')[0] # extracts 'cleaned' name (boston etc.)
    result_dict[name] = {}
    
    df = pd.read_csv(dataset, header = 0, index_col = False)
    X = df.iloc[:,0:-1]
    y = df.iloc[:,-1]
    
    default_train_results = []
    default_test_results = []
    
    optimized_train_results = []
    optimized_test_results = []
    
    time_list = []
    
    for _ in range(num_rounds):
        start_time = time.time()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

        # default model
        default_model, scoring = init_mode(mode[idx], tree_method, n_estimators)
        default_model.fit(X_train, y_train)
        
        if mode[idx] == 'regression':
            default_train = mean_squared_error(y_train, default_model.predict(X_train))
            default_test = mean_squared_error(y_test, default_model.predict(X_test))

            
        if mode[idx] == 'binary_class' or mode[idx] == 'multiple_class':
            default_train = accuracy_score(y_train, default_model.predict(X_train))
            default_test = accuracy_score(y_test, default_model.predict(X_test))
            
        # optimized model
        optimized_model = xgboost_optimizer(X_train, y_train, mode = mode[idx], n_estimators = n_estimators, verbose = 0, cv_folds = cv_folds)
        
        if mode[idx] == 'regression':
            optimized_train = mean_squared_error(y_train, optimized_model.predict(X_train))
            optimized_test = mean_squared_error(y_test, optimized_model.predict(X_test))
            
        if mode[idx] == 'binary_class' or mode[idx] == 'multiple_class':
            optimized_train = accuracy_score(y_train, optimized_model.predict(X_train))
            optimized_test = accuracy_score(y_test, optimized_model.predict(X_test))
        
        # Append and save results
        
        default_train_results.append(default_train)
        default_test_results.append(default_test)
    
        optimized_train_results.append(optimized_train)
        optimized_test_results.append(optimized_test)
        
        end_time = time.time()
        time_list.append(end_time - start_time)
        
        print("Finished round {} for {}. Average_time per round is {} minutes".format(_ + 1, name, (sum(time_list) / len(time_list)) / 60))
            
    # Save dataset results in dictionary
    result_dict[name]['default_train'] = default_train_results
    result_dict[name]['default_test'] = default_test_results
    result_dict[name]['optimized_train'] = optimized_train_results
    result_dict[name]['optimized_test'] = optimized_test_results

pickle.dump(result_dict, open('result_dict.pickle', 'wb'))

#########################
#
#   Result Tables
#
#########################

result_dict = pickle.load(open('result_dict.pickle', 'rb'))

california_housing = pd.DataFrame(result_dict['california_housing'])
print(california_housing[['default_test', 'optimized_test']].to_markdown())

digits = pd.DataFrame(result_dict['digits'])
print(digits[['default_test', 'optimized_test']].to_markdown())


