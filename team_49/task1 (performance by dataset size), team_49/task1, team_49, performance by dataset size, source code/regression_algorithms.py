# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn import metrics
from sklearn import neighbors
import util
import re
import math


def regression_pre_processing(df, name):
    X = np.array([])
    y = np.array([])

    if name == 'sum_with_noise':
        df = df.drop('Noisy Target Class', axis=1)  # dropping class

        X = df.drop(['Noisy Target'], axis=1)   # creating feature set
        y = df[['Noisy Target']]                # creating target set

    if name == 'sum_without_noise':
        df = df.drop('Target Class', axis=1)    # dropping class

        X = df.drop('Target', axis=1)           # creating feature set
        y = df[['Target']]                      # creating target set

    if name == 'twitter_buzz':
        X = df.iloc[:, :70]                     # creating feature set -- first 69 columns
        y = df.iloc[:, 77:]                     # Target set

    if name == '3d_road_network':
        X = df.iloc[:, 1:3]
        y = df.iloc[:, 3:]

    return X, y


def calculate_scores_split(regression_model, X, y):
    rm_score = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    regression_model.fit(X_train, y_train)

    y_predict = regression_model.predict(X_test)

    y_ = np.squeeze(y)
    min = y_.min()
    max = y_.max()

    mse = metrics.mean_squared_error(y_predict, y_test)
    rmse = math.sqrt(abs(mse))
    nmse = rmse/(max - min)

    exp_variance = metrics.explained_variance_score(y_predict, y_test)

    print "for nmse value:", nmse
    print "for explained_variance_score value:", exp_variance

    return {'Normalized mean square error': nmse, 'explained_variance': exp_variance}


def calculate_scores_cv(regression_model, X, y, scoring):
    y_ = np.squeeze(y)
    scores = cross_validate(regression_model, X, y_, cv=10, scoring=scoring)

    min = y_.min()
    max = y_.max()
    cv_scores = {}
    for score in scoring:
        mean_score = scores['test_' + score].mean()
        print "for", score, " value:", mean_score
        if re.match(score, "neg_mean_squared_error"):
            rmse = math.sqrt(abs(mean_score))
            nmse = rmse/(max - min)
            cv_scores['Normalized mean square error'] = nmse
        else:
            cv_scores[score]= mean_score

    return cv_scores


def regression(X, y, scoring, chunk_size, name, out_dict):

    print "Linear Regression"
    scores = calculate_scores_cv(linear_model.LinearRegression(), X, y, scoring)  # Linear Regression
    util.iterate_into_out_dict(scores, 'Linear Regression', name, out_dict, chunk_size)

    print "KNN"
    regression_model = neighbors.KNeighborsRegressor(n_neighbors=2)  # SGDRegressor

    if chunk_size < 50000:
        scores = calculate_scores_cv(regression_model, X, y, scoring)
        util.iterate_into_out_dict(scores, 'K Nearest Neighbors (2)', name, out_dict, chunk_size)
    else:
        scores = calculate_scores_split(regression_model, X, y)
        util.iterate_into_out_dict(scores, 'K Nearest Neighbors (2)', name, out_dict, chunk_size)


def perform_regression(df, name, chunk_size, out_dict):

    X, y = regression_pre_processing(df, name)

    assert X.size != 0 and y.size != 0
    scoring = ['neg_mean_squared_error', 'explained_variance']
    regression(X, y, scoring, chunk_size, name, out_dict)
