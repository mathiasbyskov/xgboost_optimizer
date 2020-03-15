"""

File to run optimize model through terminal.

It optimizes the model on the full dataset given and
dumps the optimized model to the current repository.

Using this, it is not possible to specify your own 'tree_method, 'parameters' and 'param_dict'


2 Usages:

'python run.py model_name X y mode'

'python run.py model_name X y mode n_estimators cv_folds n_jobs verbose'


"""


def main():
	""" Running the main-function. """

	import sys
	import pandas as pd
	from xgboost_optimizer import xgboost_optimizer

	args = sys.argv

	if len(args) != 5 and len(args) != 9:
		print("Number of arguments must be equal to 3 or 7 (read usage in top of file).")
		print("For 3 arguments specify: X, y and mode.")
		print("For 7 arguments spedcify: X, y, mode, n_estimators, cv_folds, n_jobs and verbose.")
		sys.exit()

	model_name = args[1]
	X = pd.read_csv(args[2], header = 0, index_col = False)
	y = pd.read_csv(args[3], header = 0, index_col = False)
	mode = args[4]
	
	if len(args) == 5:
		n_estimators = 500
		cv_folds = 5
		n_jobs = -1
		verbose = 1

	if len(args) == 9:
		n_estimators = int(args[5])
		cv_folds = int(args[6])
		n_jobs = int(args[7])
		verbose = int(args[8])

	model = xgboost_optimizer(X, y, mode = mode, n_estimators = n_estimators, cv_folds = cv_folds, verbose = verbose)

	model.save_model('{}.model'.format(model_name))


if __name__ == "__main__":
	main()