 # xgboost_optimizer

[xgboost](https://xgboost.readthedocs.io/en/latest/) has become a very popular choice of model when doing supervised learning. It contains many parameters, which makes it a difficult task to optimize/tune a good model. This library provides a automatic approach for optimizing the xgboost-model.

For a detailed explanation of the parameters to use in the function see "Important Parameters" below or the function-description in [xgboost_optimizer](https://github.com/mathiasbyskov/xgboost_optimizer/blob/master/xgboost_optimizer.py).

In order to use the library properly it is **important** that you are familiar with boosting in general and has some knowledge about the [xgboost](https://xgboost.readthedocs.io/en/latest/)-library.

## System Speciﬁcations 
For the tool to work properly, the following packages are required:

• Python 3.6 : https://anaconda.org/anaconda/python

• Pandas : https://anaconda.org/anaconda/pandas

• Scikit-learn : https://anaconda.org/anaconda/scikit-learn

• xgboost : https://xgboost.readthedocs.io/en/latest/

## Usage

The code below returns the optimized model. Two concrete examples are given in the [examples.py](https://github.com/mathiasbyskov/xgboost_optimizer/blob/master/example.py) file for both a regression and a multiple classification problem.

```python
from xgboost_optimizer import xgboost_optimizer

mode = 'regression'
tree_method = 'hist'
n_estimators = 10
cv_folds = 5
verbose = 1

X = "Your features"
y = "Target variable"

optimized_model = xgboost_optimizer(X, 
                                    y, 
                                    mode = "regression", 
                                    tree_method = tree_method,
                                    n_estimators = n_estimators, 
                                    cv_folds = cv_folds, 
                                    verbose = verbose)

# Use model to predict
optimized_model.predict(X)

```

## Usage (via terminal/powershell)

The [run.py](https://github.com/mathiasbyskov/xgboost_optimizer/blob/master/run.py) file includes an opportunity to run the optimizer-function through the terminal. The optimized model is saved in the local repository and given the "model_name.model". 

I would recommend using the upper method instead of this - since this provides more possibilities to work with the model afterwards. Although, the saved model can be loaded and used for predictions afterwards.

It can be run through the following command in the repo:

```bash

python run.py model_name X y mode

```

## Important Parameters
Many of the parameters to use in the [xgboost_optimizer](https://github.com/mathiasbyskov/xgboost_optimizer/blob/master/xgboost_optimizer.py) are similar to the ones in [xgboost](https://xgboost.readthedocs.io/en/latest/). Although, two parameters are important and fundamental for the library:

- **param_grid:**     
The param_grid is a dictionary that includes information about all the parameters, that is possible to use in the parameters-list (see below). It contains 5 different kind of information about each parameter:
    - **range**: Specifies the range, that is used in the grid for doing GridSearch. If the minimum or maximum is hit, the range is specified and another GridSearch is performed.
    - **size**: Specifies the number of elements in the grid for the parameter.
    - **dtype**: Specifies the type the parameter should be (int/float etc.)
    - **step_size**: The step_size specifies to which extend the grid should be modified if a boundary is hit (the minimum or maximum value in the range).
    - **minimum**: A boolean value that specifies whether or not the minimum-value in the grid is the lowest possible (according to the xgboost-settings).
    - **maximum**: A boolean value that specifies whether or not the maximum-value in the grid is the lowest possible (according to the xgboost-settings).

    The default param_grid is:
```python

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

```
 
- **parameters:**    
This is a list of parameters to be optimized for. For computational reasons it would be impossible to specify a grid for each parameter and do an exhaustive search.   
The list species the order and in which pairs the parameters should be optimized in. The **maximum** number of parameters to be optimized for at once is **2**.

  The default list is: 

```python
[['max_depth', 'min_child_weight'], 'gamma', ['subsample', 'colsample_bytree'], 'reg_alpha', 'learning_rate']
```



## Proof of Concept
To proof that optimization has happened after running the model I used two different datasets: [california_housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#sklearn.datasets.fetch_california_housing) for regression and [digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits) for multiple classification. The code to obtain the results are provided in [proofofconcept.py](https://github.com/mathiasbyskov/xgboost_optimizer/blob/master/proofofconcept.py).

10 different models was trained of 10 different training datasets and the MSE and Accuracy is provided in the tables below:

**california_housing (regression):**

|   Round |   Default MSE |   Optimized MSE |
|---:|---------------:|-----------------:|
|  1 |       0.213976 |         0.195635 |
|  2 |       0.200672 |         0.188292 |
|  3 |       0.221673 |         0.19783  |
|  4 |       0.212485 |         0.197299 |
|  5 |       0.21887  |         0.20354  |
|  6 |       0.21003  |         0.190388 |
|  7 |       0.224766 |         0.207012 |
|  8 |       0.20848  |         0.195121 |
|  9 |       0.203722 |         0.188492 |
|  10 |       0.216503 |         0.199135 |

**digits (multiple classification):**
|   Round |   Default Accuracy |   Optimized Accuracy|
|---:|---------------:|-----------------:|
|  1 |       0.961111 |         0.972222 |
|  2 |       0.975    |         0.980556 |
|  3 |       0.966667 |         0.988889 |
|  4 |       0.961111 |         0.972222 |
|  5 |       0.969444 |         0.972222 |
|  6 |       0.980556 |         0.980556 |
|  7 |       0.961111 |         0.977778 |
|  8 |       0.952778 |         0.975    |
|  9 |       0.963889 |         0.975    |
|  10 |       0.975    |         0.988889 |

## It runs too slow. What can i do?
xgboost can be computationally heavy to run - especially when handling high-dimensional data or very large sample-sizes. There are a few different tricks in order to make the optimizer faster - although, they should be done with care:

- Avoid pairs of parameters in parameter-list.
- n_jobs: Default is -1, which means all threads are being used in GridSearch.
- tree_method: Default is hist, which is the fastest. 
- Decrease n_estimators (could heavily decrease the performance if set too low).
- Decrease cv_folds (could heavily decrease the performance if set too low).


## Comments / Requests
Any suggestions for modifications, changes or additions are welcome! 

My personal e-mail:    
[mathias-byskov@live.dk](mailto:mathias-byskov@live.dk)