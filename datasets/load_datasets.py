
"""
Created on Mon Mar 23 21:00:40 2020

@author: Mathias Byskov Nielsen
"""


from sklearn.datasets import fetch_california_housing, load_digits
import pandas as pd

# California Housing Dataset
cal_housing = fetch_california_housing()
df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
df['target'] = cal_housing.target
df.to_csv('./datasets/california_housing.csv', header = True, index = False)

X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
X.to_csv('./datasets/california_housing_X.csv', header = True, index = False)

y = pd.DataFrame(cal_housing.target)
y.columns = ['target']
y.to_csv('./datasets/california_housing_y.csv', header = True, index = False)


# Digits Dataset
digits = load_digits()
df = pd.DataFrame(digits.data)
df['target'] = digits.target
df.to_csv('./datasets/digits.csv', header = True, index = False)