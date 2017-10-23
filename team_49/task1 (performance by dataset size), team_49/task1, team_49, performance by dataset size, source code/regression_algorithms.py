# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score


def regression_preprocessing(df, name):
    X = np.array([])
    y = np.array([])

    if name == 'sum_with_noise':
        df = df.drop('Noisy Target Class', axis=1)  # dropping class

        X = df.drop('Noisy Target', axis=1)  # creating feature set
        y = df[['Noisy Target']]  # creating target set

    return X, y


def perform_linear_regression(X, y, scoring):
    print "LinearRegression"
    regression_model = linear_model.LinearRegression()
    scores = cross_validate(regression_model, X, np.squeeze(y), cv=10, scoring=scoring)

    print scores['test_neg_mean_squared_error'].mean()
    for score in scoring:
        print(score + ": %0.2f" % (scores['test_'+score].mean()))


def perform_regression(df, name):
    X, y = regression_preprocessing(df, name)

    assert X.size != 0 and y.size != 0

    scoring = ['neg_mean_squared_error', 'r2']

    perform_linear_regression(X, y, scoring)
