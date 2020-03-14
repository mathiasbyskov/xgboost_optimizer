#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 09:39:12 2018

@author: mathias
"""


def n_estimator(alg, X_train, y_train, X_test, y_test, metric='rmse', cv_folds=5, early_stopping_rounds=50, n_estimators=5000):
    """
    Finds the optimized number of n_estimators in an XGBoost Classifier/Regressor and returns 'n_est' as an
    integer (found number of trees).

    Metric specifies whether it is a classification- or a regression-problem.

    cv_folds, early_stopping_rounds, num_boost_round (n_estimators) can be specified in the used 'xgboost.cv'-function.

    The workflow in this function was original created by Aarshay Jain.
    It can be seen in this guide:
    https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    """

    import xgboost as xgb
    from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error

    assert ((metric == 'auc') or (metric == 'rmse')), "Metric must be Root Mean Squared Error ('rmse') or Area Under the Curve ('auc')."

    # Obtaining the optimal n_estimators-value through CV
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics=metric, early_stopping_rounds=early_stopping_rounds, stratified=True)

    print(cvresult)

    alg.set_params(n_estimators=cvresult.shape[0])
    n_est = cvresult.shape[0]

    # Fit the algorithm
    alg.fit(X_train, y_train, eval_metric=metric)

    # Predict training set (Regression):
    if metric == 'rmse':
        train_pred = alg.predict(X_train)
        test_pred = alg.predict(X_test)

        # Print model report:
        print ("\nModel Report - n_estimator optimization")
        print ("RMSE Score (Train): %f" % (mean_squared_error(y_train, train_pred)) ** (0.5))
        print ("RMSE Score (Test): %f" % (mean_squared_error(y_test, test_pred)) ** (0.5))
        return n_est

    # Predict training set (Classification):
    if metric == 'auc':
        train_predprob = alg.predict_proba(X_train)[:, 1]
        test_predictions = alg.predict(X_test)

        test_predprob = alg.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, test_predictions).ravel()

        # Print model report:
        print ("\nModel Report - n_estimator optimization")
        print ("AUC Score (Train): %f" % roc_auc_score(y_train, train_predprob))
        print ("AUC Score (Test): %f" % roc_auc_score(y_test, test_predprob))
        print ("TN: {}".format(tn))
        print ("FP: {}".format(fp))
        print ("FN: {}".format(fn))
        print ("TP: {}".format(tp))
        return n_est


# ========================================================================== #
#                                                                            #
#                          REGRESSION FUNCTION  !!                           #
#                                                                            #
# ========================================================================== #


def XGBR_tuner(X, y, test_split=.2, cv_folds=5, tree_method='hist', n_jobs=4, early_stopping_rounds=50, n_estimators=5000, verbose=0):
    """


    XGBR_tuner seeks to find the optimized values for the hyperparameters in XGBRegressor().
    It returns the tuned model.

    Data

    cv_folds: The number of folds to use in all the cross-validations. Default is 5.

    tree_method: Which specific method to use in XGBRegressor(). Default is 'hist'.
        - Possible methods: exact, approx, hist, gpu_exact and gpu_hist.

    n_jobs: Number of parrallel jobs to run in XGBRegressor(). Default is 4.

    early_stopping_rounds: Number of early_stopping_rounds to use in n_estimator(). CV error needs to decrease at least every <early_stopping_rounds> round(s) to continue.
                           Default is 50.

    n_estimators: Highest number of trees to fit (the optimized number of estimators is later changed via the n_estimator()-function).
                  Default is 5000.

    verbose: The amount of messages to run each time GridSearchCV is started (0, 1 or 2). Default is 0.



    The work-flow in this function is determined by the guide made by Aarshay Jain:
    https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

    """

    from sklearn.model_selection import GridSearchCV, train_test_split
    from xgboost import XGBRegressor
    import numpy as np

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    xgb1 = XGBRegressor(learning_rate=0.1, n_estimators=n_estimators, max_depth=5, gamma=0, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
    n_estimators = n_estimator(xgb1, X_train, y_train, X_test, y_test, cv_folds=cv_folds, metric='rmse', early_stopping_rounds=early_stopping_rounds, n_estimators=n_estimators)

    # Finding optimal depth and min_child_weight
    param_test1 = {
        'max_depth': list(range(1, 7, 1)),
        'min_child_weight': list(range(1, 7, 1))
    }
    md = 0
    mcw = 0

    while (md == 0 or mcw == 0 or md == param_test1['max_depth'][5] or mcw == param_test1['min_child_weight'][5]):

        if md == param_test1['max_depth'][5]:
            del param_test1['max_depth'][0:6]

        if mcw == param_test1['min_child_weight'][5]:
            del param_test1['min_child_weight'][0:6]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
        xgb1 = XGBRegressor(learning_rate=0.1, n_estimators=n_estimators, gamma=0, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
        grid_XGboost = GridSearchCV(xgb1, param_test1, cv=cv_folds, scoring='neg_mean_squared_error', verbose=verbose).fit(X_train, y_train)

        md = grid_XGboost.best_params_.get('max_depth')
        mcw = grid_XGboost.best_params_.get('min_child_weight')

        if md == param_test1['max_depth'][5]:
            param_test1['max_depth'].extend(range(param_test1['max_depth'][5], param_test1['max_depth'][5] + 6))

        if mcw == param_test1['min_child_weight'][5]:
            param_test1['min_child_weight'].extend(range(param_test1['min_child_weight'][5], param_test1['min_child_weight'][5] + 6))

    print("GridSearch (max_depth + min_child_weight) Done! Best MSE Score {}".format(abs(grid_XGboost.best_score_) ** 0.5))

    # Finding optimal gamma
    param_test2 = {
        'gamma': [i / 10.0 for i in range(0, 11)]
    }
    gamma = -1

    while (gamma == -1 or gamma == param_test2['gamma'][10]):

        if gamma == param_test2['gamma'][10]:
            del param_test2['gamma'][0:11]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
        xgb1 = XGBRegressor(learning_rate=0.1, n_estimators=n_estimators, subsample=0.8, colsample_bytree=0.8, max_depth=md, min_child_weight=mcw, scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
        grid_XGboost = GridSearchCV(xgb1, param_test2, cv=cv_folds, scoring='neg_mean_squared_error', verbose=verbose).fit(X_train, y_train)

        gamma = grid_XGboost.best_params_.get('gamma')

        if (gamma == param_test2['gamma'][10]):
            param_test2['gamma'].extend([i / 10.0 for i in range(int(param_test2['gamma'][10] * 10), int(param_test2['gamma'][10] * 10) + 11)])

    print("GridSearch (gamma-parameter) Done! Best MSE Score {}".format(abs(grid_XGboost.best_score_) ** 0.5))

    # Reboost number of parameters

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    xgb1 = XGBRegressor(learning_rate=0.1, n_estimators=n_estimators, subsample=0.8, gamma=gamma, colsample_bytree=0.8, max_depth=md, min_child_weight=mcw, scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
    n_estimators = n_estimator(xgb1, X_train, y_train, X_test, y_test, cv_folds=cv_folds, metric='rmse', early_stopping_rounds=early_stopping_rounds, n_estimators=n_estimators)

    # Tune subsample and colsample_bytree
    param_test3 = {
        'subsample': [i / 10.0 for i in range(1, 11, 1)],
        'colsample_bytree': [i / 10.0 for i in range(1, 11, 1)]
    }
    subs = 0
    colsample = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    xgb1 = XGBRegressor(learning_rate=0.1, n_estimators=n_estimators, gamma=gamma, max_depth=md, min_child_weight=mcw, scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
    grid_XGboost = GridSearchCV(xgb1, param_test3, cv=cv_folds, scoring='neg_mean_squared_error', verbose=verbose).fit(X_train, y_train)

    subs = grid_XGboost.best_params_.get('subsample')
    colsample = grid_XGboost.best_params_.get('colsample_bytree')

    print("GridSearch (subsample + colsample_bytree) Done! Best MSE Score {}".format(abs(grid_XGboost.best_score_) ** 0.5))

    # Tune reg param
    param_test4 = {
        'reg_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 50, 100, 150, 200]
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    xgb1 = XGBRegressor(learning_rate=0.1, n_estimators=n_estimators, gamma=gamma, max_depth=md, min_child_weight=mcw, subsample=subs, colsample_bytree=colsample, scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
    grid_XGboost = GridSearchCV(xgb1, param_test4, cv=cv_folds, scoring='neg_mean_squared_error', verbose=verbose).fit(X_train, y_train)
    alpha = grid_XGboost.best_params_.get('reg_alpha')
    print("GridSearch (reg_alpha) 1/2ways Done! Best MSE Score {}".format(abs(grid_XGboost.best_score_) ** 0.5))

    param_test5 = {
        'reg_alpha': np.linspace(alpha / 2, alpha * 1.5, 20)
    }

    if alpha != 0:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
        xgb1 = XGBRegressor(learning_rate=0.1, n_estimators=n_estimators, gamma=gamma, max_depth=md, min_child_weight=mcw, colsample_bytree=colsample, subsample=subs, scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
        grid_XGboost = GridSearchCV(xgb1, param_test5, cv=cv_folds, scoring='neg_mean_squared_error', verbose=verbose).fit(X_train, y_train)
        alpha = grid_XGboost.best_params_.get('reg_alpha')

    while (alpha == param_test5['reg_alpha'][0] or alpha == param_test5['reg_alpha'][19]) and alpha != 0:

        param_test5 = {
            'reg_alpha': np.linspace(alpha / 2, alpha * 1.5, 20)
        }

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
        xgb1 = XGBRegressor(learning_rate=0.1, n_estimators=n_estimators, gamma=gamma, max_depth=md, min_child_weight=mcw, colsample_bytree=colsample, subsample=subs, scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
        grid_XGboost = GridSearchCV(xgb1, param_test5, cv=cv_folds, scoring='neg_mean_squared_error', verbose=verbose).fit(X_train, y_train)
        alpha = grid_XGboost.best_params_.get('reg_alpha')

    param_test6 = {
        'learning_rate': [i / 100 for i in range(1, 11, 1)]
    }

    lr = -1

    while (lr == -1 or lr == param_test6['learning_rate'][0] or lr == param_test6['learning_rate'][9]):

        if (lr == param_test6['learning_rate'][0] or lr == param_test6['learning_rate'][9]):
            del param_test6['learning_rate'][0:10]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
        xgb1 = XGBRegressor(n_estimators=n_estimators, subsample=0.8, colsample_bytree=0.8, max_depth=md, min_child_weight=mcw, scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
        grid_XGboost = GridSearchCV(xgb1, param_test6, cv=cv_folds, scoring='neg_mean_squared_error', verbose=verbose).fit(X_train, y_train)

        lr = grid_XGboost.best_params_.get('learning_rate')

        if lr == param_test6['learning_rate'][0]:
            param_test6['learning_rate'].extend([i / 10.0 for i in param_test6['learning_rate']])

        if lr == param_test6['learning_rate'][9]:
            param_test6['learning_rate'].extend([i * 10.0 for i in param_test6['learning_rate']])

    print("GridSearch (learning rate-parameter) Done! Best MSE Score {}".format(abs(grid_XGboost.best_score_) ** 0.5))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    n_estimators = n_estimator(xgbf, X_trian, y_train, X_test, y_test, cv_folds=cv_folds, metric='auc', early_stopping_rounds=early_stopping_rounds, n_estimators=n_estimators)

    # Set up tuned model
    print("NOW: FITTING FINAL MODEL - WITH TUNED PARAMETERS")
    xgbf = XGBRegressor(learning_rate=lr,
                        n_estimators=n_estimators,
                        max_depth=md,
                        min_child_weight=mcw,
                        gamma=gamma,
                        subsample=subs,
                        colsample_bytree=colsample,
                        reg_alpha=alpha,
                        tree_method='exact',
                        scale_pos_weight=1,
                        n_jobs=n_jobs,
                        silent=True)

    model = xgbf.fit(X_train, y_train)

    return model


# ========================================================================== #
#                                                                            #
#                          CLASSIFIER FUNCTION  !!                           #
#                                                                            #
# ========================================================================== #


def XGBC_tuner(X, y, test_split=.2, cv_folds=5, tree_method='hist', n_jobs=4, early_stopping_rounds=5, n_estimators=5000, verbose=0):
    """

    XGBC_tuner seeks to find the optimized values for the hyperparameters in XGBClassifier().
    It returns the tuned model.

    X_train, X_test: Training- and test-data set in pd.DataFrame-format. Only features should be included.

    y_train, y_test: Training and test-data in pd.Series-format. A list of all the responses.

    cv_folds: The number of folds to use in all the cross-validations. Default is 5.

    tree_method: Which specific method to use in XGBRegressor(). Default is 'hist'.
        - Possible methods: exact, approx, hist, gpu_exact and gpu_hist.

    n_jobs: Number of parrallel jobs to run in XGBRegressor(). Default is 4.

    early_stopping_rounds: Number of early_stopping_rounds to use in n_estimator(). CV error needs to decrease at least every <early_stopping_rounds> round(s) to continue.
                           Default is 50.

    n_estimators: Highest number of trees to fit (the optimized number of estimators is later changed via the n_estimator()-function).
                  Default is 5000.

    verbose: The amount of messages to run each time GridSearchCV is started (0, 1 or 2). Default is 0.

    The work-flow in this function is determined by the guide made by Aarshay Jain:
    https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

    """

    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    import numpy as np
    #n_estimators = 300
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=n_estimators, max_depth=2, min_child_weight=1, gamma=0, subsample=0, colsample_bytree=0, objective='binary:logistic', scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
    n_estimators = n_estimator(xgb1, X_train, y_train, X_test, y_test, cv_folds=cv_folds, metric='auc', early_stopping_rounds=early_stopping_rounds, n_estimators=n_estimators)
    print("n_est: {}".format(n_estimators))
    # Finding optimal depth and min_child_weight
    param_test1 = {
        'max_depth': list(range(1, 7, 1)),
        'min_child_weight': list(range(1, 7, 1))
    }

    md = 0
    mcw = 0
    while (md == 0 or mcw == 0 or md == param_test1['max_depth'][5] or mcw == param_test1['min_child_weight'][5]):

        if md == param_test1['max_depth'][5]:
            del param_test1['max_depth'][0:6]

        if mcw == param_test1['min_child_weight'][5]:
            del param_test1['min_child_weight'][0:6]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
        xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=n_estimators, gamma=0, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
        grid_XGboost = GridSearchCV(xgb1, param_test1, cv=cv_folds, scoring='roc_auc', verbose=verbose).fit(X_train, y_train)

        md = grid_XGboost.best_params_.get('max_depth')
        mcw = grid_XGboost.best_params_.get('min_child_weight')
        print(md, mcw)

        if md == param_test1['max_depth'][5]:
            param_test1['max_depth'].extend(range(param_test1['max_depth'][5], param_test1['max_depth'][5] + 6))

        if mcw == param_test1['min_child_weight'][5]:
            param_test1['min_child_weight'].extend(range(param_test1['min_child_weight'][5], param_test1['min_child_weight'][5] + 6))
        print(param_test1)

    print("GridSearch (max_depth + min_child_weight) Done! Best AUC Score {}".format(grid_XGboost.best_score_))

    # Finding optimal gamma
    param_test2 = {
        'gamma': [i / 10.0 for i in range(0, 11)]
    }
    gamma = -1

    while (gamma == -1 or gamma == param_test2['gamma'][10]):

        if gamma == param_test2['gamma'][10]:
            del param_test2['gamma'][0:11]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
        xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=n_estimators, max_depth=md, min_child_weight=mcw, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
        grid_XGboost = GridSearchCV(xgb1, param_test2, cv=cv_folds, scoring='roc_auc', verbose=verbose).fit(X_train, y_train)

        gamma = grid_XGboost.best_params_.get('gamma')

        if (gamma == param_test2['gamma'][10]):
            param_test2['gamma'].extend([i / 10.0 for i in range(int(param_test2['gamma'][10] * 10), int(param_test2['gamma'][10] * 10) + 11)])

    print("GridSearch (gamma) Done! Best AUC Score {}".format(grid_XGboost.best_score_))

    # Reboost number of parameters

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=n_estimators, subsample=0.8, gamma=gamma, colsample_bytree=0.8, max_depth=md, min_child_weight=mcw, objective='binary:logistic', scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
    n_estimators = n_estimator(xgb1, X_train, y_train, X_test, y_test, cv_folds=cv_folds, metric='auc', early_stopping_rounds=early_stopping_rounds, n_estimators=n_estimators)
    print("n_est: {}".format(n_estimators))
    # Tune subsample and colsample_bytree
    param_test3 = {
        'subsample': [i / 10.0 for i in range(1, 11, 1)],
        'colsample_bytree': [i / 10.0 for i in range(1, 11, 1)]
    }
    subs = 0
    colsample = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=n_estimators, gamma=gamma, max_depth=md, min_child_weight=mcw, objective='binary:logistic', scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
    grid_XGboost = GridSearchCV(xgb1, param_test3, cv=cv_folds, scoring='roc_auc', verbose=verbose).fit(X_train, y_train)

    subs = grid_XGboost.best_params_.get('subsample')
    colsample = grid_XGboost.best_params_.get('colsample_bytree')

    print("GridSearch (subsample + colsample_bytree) Done! Best AUC Score {}".format(grid_XGboost.best_score_))

    # Tune reg param
    param_test4 = {
        'reg_alpha': [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 50, 100, 150, 200]
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=n_estimators, gamma=gamma, max_depth=md, min_child_weight=mcw, colsample_bytree=colsample, subsample=subs, objective='binary:logistic', scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
    grid_XGboost = GridSearchCV(xgb1, param_test4, cv=cv_folds, scoring='roc_auc', verbose=verbose).fit(X_train, y_train)
    alpha = grid_XGboost.best_params_.get('reg_alpha')
    print("GridSearch (reg_alpha) 1/2 ways Done! Best AUC Score {}".format(grid_XGboost.best_score_))

    param_test5 = {
        'reg_alpha': np.linspace(alpha / 2, alpha * 1.5, 20)
    }

    if alpha != 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
        xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=n_estimators, gamma=gamma, max_depth=md, min_child_weight=mcw, colsample_bytree=colsample, subsample=subs, objective='binary:logistic', scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
        grid_XGboost = GridSearchCV(xgb1, param_test5, cv=cv_folds, scoring='roc_auc', verbose=verbose).fit(X_train, y_train)
        alpha = grid_XGboost.best_params_.get('reg_alpha')

    while (alpha == param_test5['reg_alpha'][0] or alpha == param_test5['reg_alpha'][19]) and alpha != 0:
        param_test5 = {
            'reg_alpha': np.linspace(alpha / 2, alpha * 1.5, 20)
        }
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
        xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=n_estimators, gamma=gamma, max_depth=md, min_child_weight=mcw, colsample_bytree=colsample, subsample=subs, objective='binary:logistic', scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs)
        grid_XGboost = GridSearchCV(xgb1, param_test5, cv=cv_folds, scoring='roc_auc', verbose=verbose).fit(X_train, y_train)
        alpha = grid_XGboost.best_params_.get('reg_alpha')

    print("GridSearch (reg_alpha) Done! Best AUC Score {}".format(grid_XGboost.best_score_))

    # Reboost parameters!

    # Finding optimal learning-rate
    param_test6 = {
        'learning_rate': [i / 100 for i in range(1, 11, 1)]
    }

    lr = -1

    while (lr == -1 or lr == param_test6['learning_rate'][0] or lr == param_test6['learning_rate'][9]):

        if (lr == param_test6['learning_rate'][0] or lr == param_test6['learning_rate'][9]):
            del param_test6['learning_rate'][0:10]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
        xgb1 = XGBClassifier(n_estimators=n_estimators, gamma=gamma, max_depth=md, min_child_weight=mcw, colsample_bytree=colsample, subsample=subs, objective='binary:logistic', scale_pos_weight=1, tree_method=tree_method, n_jobs=n_jobs, alpha=alpha)
        grid_XGboost = GridSearchCV(xgb1, param_test6, cv=cv_folds, scoring='roc_auc', verbose=verbose).fit(X_train, y_train)

        lr = grid_XGboost.best_params_.get('learning_rate')

        print("LR: {}".format(lr))
        print(param_test6['learning_rate'])
        print(abs(grid_XGboost.best_score_))

        if lr == param_test6['learning_rate'][0]:
            param_test6['learning_rate'].extend([i / 10.0 for i in param_test6['learning_rate']])

        if lr == param_test6['learning_rate'][9]:
            param_test6['learning_rate'].extend([i * 10.0 for i in param_test6['learning_rate']])

    print("GridSearch (learning rate-parameter) Done! Best AUC Score {}".format(abs(grid_XGboost.best_score_)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    n_estimators = n_estimator(xgb1, X_train, y_train, X_test, y_test, cv_folds=cv_folds, metric='auc', early_stopping_rounds=early_stopping_rounds, n_estimators=n_estimators)
    print("n_est: {}".format(n_estimators))
    # Assessment of predictions a good model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    xgbf = XGBClassifier(learning_rate=lr,
                         n_estimators=n_estimators,
                         max_depth=md,
                         min_child_weight=mcw,
                         gamma=gamma,
                         subsample=subs,
                         colsample_bytree=colsample,
                         reg_alpha=alpha,
                         objective='binary:logistic',
                         tree_method='exact',
                         scale_pos_weight=1)

    model = xgbf.fit(X_train, y_train)
    test_predprob = model.predict_proba(X_test)[:, 1]
    print(roc_auc_score(y_test, test_predprob))

    return model
