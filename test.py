# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 04:00:00 2020

@author: mathi
"""


#%%


#%%
    
# BOSTON

import os
os.chdir('C:/Users/mathi/Desktop/xgboost_optimizer/')
from xgboost_optimizer import xgboost_optimizer

import pandas as pd
os.chdir('C:/Users/mathi/Desktop/xgboost_optimizer/datasets/')

boston = pd.read_csv('boston.csv', header = 0, index_col = False)  

X = boston.iloc[:,0:-1]
y = boston.iloc[:,-1]


final_param = xgboost_optimizer(X, y, verbose = 2)
#%%

# DIABETES

import os
import pandas as pd
os.chdir('C:/Users/mathi/Desktop/xgboost_optimizer/datasets/')

diabetes = pd.read_csv('diabetes.csv', header = 0, index_col = False)  

X = diabetes.iloc[:,0:-1]
y = diabetes.iloc[:,-1]

final_param = xgboost_optimizer(X, y)

#%%

# BREAST CANCER (BINARY)

import os
import pandas as pd
os.chdir('C:/Users/mathi/Desktop/xgboost_optimizer/datasets/')

breast_cancer = pd.read_csv('breast_cancer.csv', header = 0, index_col = False)  

X = breast_cancer.iloc[:,0:-1]
y = breast_cancer.iloc[:,-1]

final_param = xgboost_optimizer(X, y, mode = 'binary_class')

#%%

# WINE (3 CLASSES)

import os
import pandas as pd
os.chdir('C:/Users/mathi/Desktop/xgboost_optimizer/datasets/')

wine = pd.read_csv('wine.csv', header = 0, index_col = False)  

X = wine.iloc[:,0:-1]
y = wine.iloc[:,-1]

final_param = xgboost_optimizer(X, y, mode = 'multiple_class')

#%%
final_param = xgboost_optimizer(X, y)

