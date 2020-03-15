
"""

	LOAD ALL DATASETS

	This script loads and saves all 
	the relevant datasets.

"""

import pandas as pd
from sklearn import datasets

# BOSTON  
boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['PRICE'] = boston.target
df.to_csv('boston.csv', header=True, index=False)

# dump X and y for Boston
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

X.to_csv('X_boston.csv', header=True, index=False)
y.to_csv('y_boston.csv', header=True, index=False)

# DIABETES
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['progression'] = diabetes.target
df.to_csv('diabetes.csv', header = True, index = False)

# WINE
data = datasets.load_wine()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['wine_class'] = data.target
wine_class = {0: 'class_0', 1: 'class_1', 2:'class_2'}
df.wine_class = [wine_class[item] for item in df.wine_class]
df.to_csv('wine.csv', header = True, index = False)

# BREAST CANCER
data = datasets.load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['cancer'] = data.target
cancer = {0: 'malignant', 1: 'benign'}
df.cancer = [cancer[item] for item in df.cancer]
df.to_csv('breast_cancer.csv', header=True, index=False)