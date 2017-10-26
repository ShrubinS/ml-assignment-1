# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn import metrics
from sklearn import neighbors
from sklearn import tree
import util


def create_label_utility(target, first, second):
    if target < first:
        return 0
    elif first <= target < second:
        return 1
    else:
        return 2


def classification_pre_processing(df, name):
    X = np.array([])
    y = np.array([])

    if name == 'sum_with_noise':
        # Label encoding
        df['Noisy Target Class (Encoded)'] = df['Noisy Target Class'].astype('category')
        df['Noisy Target Class_codes'] = df['Noisy Target Class (Encoded)'].cat.codes

        X = df.drop(['Noisy Target', 'Noisy Target Class', 'Noisy Target Class (Encoded)', 'Noisy Target Class_codes'],
                    axis=1)
        y = df[['Noisy Target Class_codes']]

    if name == 'sum_without_noise':
        df['Target Class (Encoded)'] = df['Target Class'].astype('category')
        df['Target Class_codes'] = df['Target Class (Encoded)'].cat.codes

        X = df.drop(['Target', 'Target Class', 'Target Class (Encoded)', 'Target Class_codes'],
                    axis=1)
        y = df[['Target Class_codes']]          # creating target set

    if name == 'twitter_buzz':
        X = df.iloc[:, :70]                     # creating feature set -- first 69 columns

        # Create labels
        target_column = 77
        y_ = df[target_column]

        # Labels created for values of each specified quantile (eg. 1/3rd is low, 1/3rd medium, 1/3rd high)
        first = y_.quantile(.33)
        second = y_.quantile(.67)
        y = df.apply(lambda row: create_label_utility(row[target_column], first, second), axis=1)   # Target set

    if name == '3d_road_network':
        X = df.iloc[:, 1:3]                     # creating feature set -- using columns 1 and 2; not using 0th

        # Create labels
        target_column = 3
        y_ = df[target_column]

        # Labels created for values of each specified quantile (eg. 1/3rd is low, 1/3rd medium, 1/3rd high)
        first = y_.quantile(.33)
        second = y_.quantile(.67)
        y = df.apply(lambda row: create_label_utility(row[target_column], first, second), axis=1)  # Target set

    return X, y


def calculate_scores_split(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_predict, y_test)
    f1_macro = metrics.f1_score(y_predict, y_test, average='macro')
    print "for accuracy:", accuracy
    print "for f1_score value:", f1_macro

    return {'accuracy': accuracy, 'f1_macro': f1_macro}


def calculate_scores_cv(regression_model, X, y, scoring):
    scores = cross_validate(regression_model, X, np.squeeze(y), cv=10, scoring=scoring)

    cv_scores = {}

    for score in scoring:
        mean_score = scores['test_' + score].mean()
        print "for", score, " value:", mean_score
        cv_scores[score]= mean_score

    return cv_scores


def classification(X, y, scoring, chunk_size, name, out_dict):

    print "Logistic Regression"
    scores = calculate_scores_cv(linear_model.LogisticRegression(), X, y, scoring)       # Logistic Regression
    util.iterate_into_out_dict(scores, 'Logisitc Regression', name, out_dict, chunk_size)

    print "Decision tree classifier"
    regression_model = tree.DecisionTreeClassifier()                            # Decision Tree Classifier

    if chunk_size < 50000:
        scores = calculate_scores_cv(regression_model, X, y, scoring)
        util.iterate_into_out_dict(scores, 'Decision Tree Classifier', name, out_dict, chunk_size)
    else:
        scores = calculate_scores_split(regression_model, X, y)
        util.iterate_into_out_dict(scores, 'Decision Tree Classifier', name, out_dict, chunk_size)


def perform_classification(df, name, chunk_size, out_dict):
    X, y = classification_pre_processing(df, name)

    assert X.size != 0 and y.size != 0

    # recallScorer = make_scorer(metrics.recall_score, average=None)
    # accuracyScorer = make_scorer(metrics.accuracy_score)

    scoring = ['accuracy', 'f1_macro']
    # 'neg_log_loss',

    classification(X, y, scoring, chunk_size, name, out_dict)
