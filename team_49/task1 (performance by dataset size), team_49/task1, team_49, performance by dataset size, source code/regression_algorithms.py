# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn import metrics
from sklearn import neighbors


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    regression_model.fit(X_train, y_train)

    y_predict = regression_model.predict(X_test)
    print "for mean_squared_error value:", metrics.mean_squared_error(y_predict, y_test)
    print "for explained_variance_score value:", metrics.explained_variance_score(y_predict, y_test)


def calculate_scores_cv(regression_model, X, y, scoring):
    scores = cross_validate(regression_model, X, np.squeeze(y), cv=10, scoring=scoring)

    for score in scoring:
        print "for", score, " value:", scores['test_' + score].mean()


def regression(X, y, scoring, chunk_size):

    print "Linear Regression"
    calculate_scores_cv(linear_model.LinearRegression(), X, y, scoring)    # Linear Regression

    print "KNN"
    regression_model = neighbors.KNeighborsRegressor(n_neighbors=2)  # SGDRegressor

    if chunk_size < 50000:
        calculate_scores_cv(regression_model, X, y, scoring)
    else:
        calculate_scores_split(regression_model, X, y)
    print "\n"


def perform_regression(df, name, chunk_size):
    X, y = regression_pre_processing(df, name)

    assert X.size != 0 and y.size != 0

    scoring = ['neg_mean_squared_error', 'explained_variance']

    regression(X, y, scoring, chunk_size)
