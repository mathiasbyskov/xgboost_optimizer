

from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor, XGBClassifier
import numpy as np


class XGBtuner:

	def __init__(self, X, y):
		self.features = X
		self.preidctors = y
		self.scoredict = {}


	def one_parameter(self):
		pass

	def two_parameter(self):
		pass

	def tuner(self):
		pass

	def predict(self):
		pass




#if __name__ == "__main__":
#	XGBtuner()

"""
XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, 
	verbosity=1, objective='binary:logistic', booster='gbtree', 
	tree_method='auto', n_jobs=1, gpu_id=-1, gamma=0, 
	min_child_weight=1, max_delta_step=0, subsample=1, 
	colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, 
	reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
	random_state=0, missing=None, **kwargs)

XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, 
	verbosity=1, objective='reg:squarederror', booster='gbtree', 
	tree_method='auto', n_jobs=1, gamma=0, min_child_weight=1, 
	max_delta_step=0, subsample=1, colsample_bytree=1, 
	colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, 
	reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
	random_state=0, missing=None, num_parallel_tree=1, 
	importance_type='gain', **kwargs)


param_dict = {'max_depth', 'min_child_weight', 
'subsample',
'colsample_bytree', 
'reg_alpha', 
'gamma',
'learning_rate']

n_estimators?
verbosity
objective
booster
tree_method
n_jobs
"""