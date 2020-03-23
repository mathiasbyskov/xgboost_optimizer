
"""
Created on Sat Mar 14 01:52:14 2019
@author: Mathias Byskov Nielsen

This file includes all necessary functions realted to optimizing a XGBooster.
All functions

"""


########################################################
#
#    Helper Functions (for the 2 next sections)
#
########################################################

# parameter_optimization (one + two parameter optimization)

def init_parameter(parameter, param_dict):
    """ 
    Helper function: Initializes all informations for a parameter. 
    
    """
    size = param_dict[parameter]['size']
    
    step_size = param_dict[parameter]['step_size']
    dtype = param_dict[parameter]['dtype']
    minimum = param_dict[parameter]['minimum']
    maximum = param_dict[parameter]['maximum']
    
    return size, step_size, dtype, minimum, maximum

def init_param_grid(parameter, param_dict):
	""" Helper function: Initializes the parameter_grid to use in GridSearchCV. """
	
	import numpy as np

	if type(parameter) == str: # one parameter

		param_grid = {parameter : list(np.linspace(start = param_dict[parameter]['range'][0], 
												   stop  = param_dict[parameter]['range'][1], 
												   num   = param_dict[parameter]['size'], 
												   dtype = param_dict[parameter]['dtype']))
					  }

	if type(parameter) == list: # two parameters
		
		first_parameter = parameter[0]
		second_parameter = parameter[1]

		param_grid = {first_parameter : list(np.linspace(start  = param_dict[first_parameter]['range'][0], 
														 stop   = param_dict[first_parameter]['range'][1], 
														 num    = param_dict[first_parameter]['size'], 
														 dtype  = param_dict[first_parameter]['dtype'])),

                  	  second_parameter : list(np.linspace(start = param_dict[second_parameter]['range'][0], 
                  	  									  stop  = param_dict[second_parameter]['range'][1], 
                  	  									  num   = param_dict[second_parameter]['size'], 
												   		  dtype = param_dict[second_parameter]['dtype']))
                  	  }

	return param_grid

def get_conditions(parameter, best_param_val, param_grid, minimum, maximum):
    """ Helper function: Extracts conditions for a parameter. Determines whether the while-loop will continue or break. """
    
    COND_MIN = (best_param_val == param_grid[parameter][0]) and (not minimum)
    COND_MAX = (best_param_val == param_grid[parameter][-1]) and (not maximum)
    
    return COND_MIN, COND_MAX

def update_param_grid(parameter, param_grid, step_size, size, dtype, transform):
	""" Helper function: Updates param_grid according to the step_size and size from param_dict. """

	import numpy as np

	if transform == 'decrease':
		return list(np.linspace(param_grid[parameter][0] / step_size, param_grid[parameter][0], num = size, dtype = dtype)) 

	if transform == 'increase':
		return list(np.linspace(param_grid[parameter][-1], param_grid[parameter][-1] * step_size, num = size, dtype = dtype))

def set_verbose(verbose):
    """ Helper function: Sets the correct verbosity for the gridsearch (should be 1, if overall verbose is 2), """
    
    if verbose == 2:
        verbose_grid = 1
    else:
        verbose_grid = 0
        
    return verbose_grid

# xgboost_optimizer

def init_param_dict():
	""" Helper function: Initializes the default param_grid. """

	param_dict = {'max_depth': {'range': [1,7], 'size': 7, 'step_size': 2, 'dtype': int, 'minimum': True, 'maximum': False},
                  'min_child_weight': {'range': [0,6], 'size': 7, 'step_size': 2, 'dtype': int, 'minimum': True, 'maximum': False},
                  'max_delta_step': {'range': [0,6], 'size': 6, 'step_size': 2, 'dtype': int, 'minimum': True, 'maximum': False},
                  'subsample': {'range': [0.1, 1], 'size': 10, 'step_size': None, 'dtype': float, 'minimum': True, 'maximum': True},
                  'colsample_bytree': {'range': [0, 1], 'size': 10, 'step_size': None, 'dtype': float, 'minimum': True, 'maximum': True},
                  'colsample_bylevel': {'range': [0.1, 1], 'size': 10, 'step_size': None, 'dtype': float, 'minimum': True, 'maximum': True},
                  'colsample_bynode': {'range': [0.1, 1], 'size': 10, 'step_size': None, 'dtype': float, 'minimum': True, 'maximum': True},
                  'reg_alpha': {'range': [0, 0.5], 'size': 10, 'step_size': 5, 'dtype': float, 'minimum': True, 'maximum': False},
                  'reg_lambda': {'range': [0, 0.5], 'size': 10, 'step_size': 5, 'dtype': float, 'minimum': True, 'maximum': False},
                  'gamma': {'range': [0, 1.0], 'size': 11, 'step_size': 2, 'dtype': float, 'minimum': True, 'maximum': False},
                  'learning_rate': {'range': [0.001, 0.1], 'size': 10, 'step_size': 2, 'dtype': float, 'minimum': False, 'maximum': False}}
	
	return param_dict
	
def init_param_list():
	""" Helper function: Initializes the default param_list """
	return [['max_depth', 'min_child_weight'], 'gamma', ['subsample', 'colsample_bytree'], 'reg_alpha', 'learning_rate']

def init_mode(mode, tree_method, n_estimators):
    """ Helper function: Initializes the mode of the xgboost_optimizer (regression/binary class./multiple class.)"""
    
    import xgboost as xgb
    
    if mode == "binary_class":
        model = xgb.XGBClassifier(objective="binary:logistic", tree_method = tree_method, n_estimators = n_estimators)
        scoring = 'accuracy'

    if mode == "multiple_class":
        model = xgb.XGBClassifier(objective="multi:softprob", tree_method = tree_method, n_estimators = n_estimators)
        scoring = 'accuracy'
        
    if mode == "regression":
        model = xgb.XGBRegressor(objective='reg:squarederror', tree_method = tree_method, n_estimators = n_estimators)
        scoring = 'neg_mean_squared_error'
        
    return model, scoring


########################################################
#
#  Functions for optimizing for one or two parameters
#
########################################################

def one_parameter(X, y, parameter, param_dict, model, scoring, cv_folds = 5, n_jobs = -1, verbose = 0):
    """
    Takes the dataset and optimizes for one parameter using GridSearchCV (sklearn).
    
    Input:
        X: pandas dataset with all features
        y: pandas Series with all labels
        parameter: Specific parameter that will be optimized (string format)
        param_dict: dictionary with info about all features (see xgboost_optimizer for explanation)
        model: xgboost model (with pre-specified parameters)
        scoring: scoring (either area under curve (classification) or mean squared error)
        cv_folds: number of folds used in grid-search
        n_jobs: number of parallel jobs to run in grid-search
        verbose: verbosity of grid-serch
    
    Output:
        parameter: parameter that is optimized for (string)
        best_param_val: optimized value for parameter (float)
        
    """
    
    from sklearn.model_selection import GridSearchCV
    
    verbose_grid = set_verbose(verbose)
    
    param_grid = init_param_grid(parameter, param_dict)
    
    size, step_size, dtype, minimum, maximum = init_parameter(parameter, param_dict)    
    best_param_val = -1       
    COND_MIN, COND_MAX = get_conditions(parameter, best_param_val, param_grid, minimum, maximum)

        
    while (COND_MIN or COND_MAX or best_param_val == -1):
            
        grid_search = GridSearchCV(model, param_grid, scoring = scoring, cv = cv_folds, n_jobs = n_jobs, verbose = verbose_grid).fit(X, y)
        best_param_val = grid_search.best_params_[parameter]

        COND_MIN, COND_MAX = get_conditions(parameter, best_param_val, param_grid, minimum, maximum)

        if COND_MIN:
            transform = "decrease"
            param_grid[parameter] = update_param_grid(parameter, param_grid, step_size, size, dtype, transform = transform) 
            maximum = True
            continue
            
        if COND_MAX:
            transform = "increase"
            param_grid[parameter] = update_param_grid(parameter, param_grid, step_size, size, dtype, transform = transform)
            minimum = True
            continue
        
        
    if verbose: print(" {} optimizied to value: {}.".format(parameter, best_param_val))
    if verbose: print(" {} value: {}\n".format(scoring, grid_search.best_score_))
    
    return parameter, best_param_val

def two_parameters(X, y, parameter, param_dict, model, scoring, cv_folds = 5, n_jobs = -1, verbose = 0):
    """
        Takes the dataset and optimizes for two parameters using the GridSearchCV from sklearn.
    
    Input:
        X: pandas dataset with all features
        y: pandas Series with all labels
        parameter: Specific parameters that will be optimized (list format)
        param_dict: dictionary with info about all features (see xgboost_optimizer for explanation)
        model: xgboost model (with pre-specified parameters)
        scoring: scoring (either area under curve (classification) or mean squared error)
        cv_folds: number of folds used in grid-search
        n_jobs: number of parallel jobs to run in grid-search
        verbose: verbosity of grid-serch
    
    Output:
        first_parameter: first parameter that is optimized for (string)
        best_param_val_first: first optimized value for parameter (float)
        second_parameter: second parameter that is optimized for (string)
        best_param_val_second: second optimized value for parameter (float)
        
    """
    
    from sklearn.model_selection import GridSearchCV
    
    verbose_grid = set_verbose(verbose)
    
    param_grid = init_param_grid(parameter, param_dict)
    
    first_parameter = parameter[0]
    first_size, first_step_size, first_dtype, first_minimum, first_maximum = init_parameter(first_parameter, param_dict)
    best_param_val_first = -1
    COND_MIN_FIRST, COND_MAX_FIRST = get_conditions(first_parameter, best_param_val_first, param_grid, first_minimum, first_maximum)
    
    second_parameter = parameter[1]
    second_size, second_step_size, second_dtype, second_minimum, second_maximum = init_parameter(second_parameter, param_dict)
    best_param_val_second = -1
    COND_MIN_SECOND, COND_MAX_SECOND = get_conditions(second_parameter, best_param_val_second, param_grid, second_minimum, second_maximum)
        
    while (COND_MIN_FIRST or COND_MAX_FIRST or COND_MIN_SECOND or COND_MAX_SECOND or best_param_val_first == -1 or best_param_val_second == -1):
        
        grid_search = GridSearchCV(model, param_grid, scoring = scoring, cv = cv_folds, n_jobs = n_jobs, verbose = verbose_grid).fit(X, y)
            
        best_param_val_first = grid_search.best_params_[first_parameter]
        COND_MIN_FIRST, COND_MAX_FIRST = get_conditions(first_parameter, best_param_val_first, param_grid, first_minimum, first_maximum)
        
        best_param_val_second = grid_search.best_params_[second_parameter]
        COND_MIN_SECOND, COND_MAX_SECOND = get_conditions(second_parameter, best_param_val_second, param_grid, second_minimum, second_maximum)

        if COND_MIN_FIRST:
            transform = 'decrease'
            param_grid[first_parameter] = update_param_grid(first_parameter, param_grid, first_step_size, first_size, first_dtype, transform = transform) 
            first_maximum = True
            continue
            
        if COND_MAX_FIRST:
            transform = 'increase'
            param_grid[first_parameter] = update_param_grid(first_parameter, param_grid, first_step_size, first_size, first_dtype, transform = transform)
            first_minimum = True
            continue
            
        if COND_MIN_SECOND:
            transform = 'decrease'
            param_grid[second_parameter] = update_param_grid(second_parameter, param_grid, second_step_size, second_size, second_dtype, transform = transform)
            second_maximum = True
            continue
            
        if COND_MAX_SECOND:
            transform = 'increase'
            param_grid[second_parameter] = update_param_grid(second_parameter, param_grid, second_step_size, second_size, second_dtype, transform = transform)
            second_minimum = True
            continue
        
    if verbose: print(" {} optimizied to value: {}.".format(first_parameter, best_param_val_first))
    if verbose: print(" {} optimizied to value: {}.".format(second_parameter, best_param_val_second))
    if verbose: print(" {} value: {}\n".format(scoring, grid_search.best_score_))
    
    return first_parameter, best_param_val_first, second_parameter, best_param_val_second

########################################################
#
#  Final function: Uses the two functions above to 
#                  optimize the XGboost model
#
########################################################

def xgboost_optimizer(X, y, mode = "regression", parameters = None, param_dict = None, tree_method = 'hist', n_estimators = 500, cv_folds = 5, n_jobs = -1, verbose = 1):
    """
    Takes a set of features and labels and optimizes the model.
    Dumps the model to the specified path.
    
    Input:
        X: pandas dataset with all features
        y: pandas Series with all labels
        mode: Either "regression" or "classification" to choose which xgboost-function to use
        parameters: List with parameters to optimize. The order of the list determines the order of parameters to optimise.
                    Default: [['max_depth', 'min_child_weight'], 'gamma', ['subsample', 'colsample_bytree'], 'reg_alpha', 'learning_rate']
        param_dict: Dictionary with all features possible to optimize. Contains informatino about range, size, step-size, maximum and minimum value
                    Default: param_dict = {'max_depth': {'range': [1,7], 'size': 6, 'step_size': 2, 'dtype': int, 'minimum': True, 'maximum': False},
                                           'min_child_weight': {'range': [0,6], 'size': 6, 'step_size': 2, 'dtype': int, 'minimum': True, 'maximum': False},
                                           'max_delta_step': {'range': [0,6], 'size': 6, 'step_size': 2, 'dtype': int, 'minimum': True, 'maximum': False},
                                           'subsample': {'range': [0.1, 1], 'size': 10, 'step_size': None, 'dtype': float, 'minimum': True, 'maximum': True},
                                           'colsample_bytree': {'range': [0, 1], 'size': 10, 'step_size': None, 'dtype': float, 'minimum': True, 'maximum': True},
                                           'colsample_bylevel': {'range': [0.1, 1], 'size': 10, 'step_size': None, 'dtype': float, 'minimum': True, 'maximum': True},
                                           'colsample_bynode': {'range': [0.1, 1], 'size': 10, 'step_size': None, 'dtype': float, 'minimum': True, 'maximum': True},
                                           'reg_alpha': {'range': [0, 0.5], 'size': 10, 'step_size': 5, 'dtype': float, 'minimum': True, 'maximum': False},
                                           'reg_lambda': {'range': [0, 0.5], 'size': 10, 'step_size': 5, 'dtype': float, 'minimum': True, 'maximum': False},
                                           'gamma': {'range': [0, 1.0], 'size': 11, 'step_size': 2, 'dtype': float, 'minimum': True, 'maximum': False},
                                           'learning_rate': {'range': [0.001, 0.1], 'size': 10, 'step_size': 2, 'dtype': float, 'minimum': False, 'maximum': False}}
        tree_method: method to use when creating trees (parameter in xgboost-function)
        n_estimators: number of estimators to use
        cv_folds: number of folds used in grid-search
        n_jobs: number of parallel jobs to run in grid-search (-1 indicates all available threads)
        verbose: verbosity (0 nothing will be printed, 1 xgboost_optimizer will print info., 2 Gridsearch will also print info.)
        
    Output:
        model: optimized model
    """
    
    if not param_dict:
        param_dict = init_param_dict()
    
    if not parameters:
        parameters = init_param_list()  
    
    model, scoring = init_mode(mode, tree_method, n_estimators)

    
    model_parameters = model.get_params()
    
    if verbose: print("\n |---- Model Optimization Begun ----| \n")
    
    for parameter in parameters:
        if verbose: print(" Optimizing for {}.\n".format(parameter))
        
        if type(parameter) == str: # One parameter is optimized here
            
            parameter, best_param_val = one_parameter(X, y, parameter, param_dict, model, scoring, 
                                                      cv_folds = cv_folds, n_jobs = n_jobs, verbose = verbose)
            model_parameters[parameter] = best_param_val
            model.set_params(**model_parameters)
        
        if type(parameter) == list: # Two parameters are optimized here
            
            first_parameter, best_param_val_first, second_parameter, best_param_val_second = two_parameters(X, y, parameter, param_dict, model, 
                                                                                                            scoring, cv_folds = cv_folds, 
                                                                                                            n_jobs = n_jobs, verbose = verbose)
            model_parameters[first_parameter] = best_param_val_first
            model_parameters[second_parameter] = best_param_val_second
            model.set_params(**model_parameters)
            
            
    model = model.fit(X, y)
    if verbose: print("\nFinal model optimized.")
    return model


