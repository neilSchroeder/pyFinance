import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from collections import Counter
from sklearn import svm, neighbors, model_selection
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import class_mlProfile


def machineLearn(ticker):
    """get features from data"""
    this_mlProfile = class_mlProfile.mlProfile(ticker, 7)
    drop_cols = []
    keep_cols = []
    for colName, col in this_mlProfile.df.iteritems():
        if colName.find(ticker) == -1:
            drop_cols.append(colName)
        else:
            if colName.find('_x') != -1 and colName.find('_y') != -1:
                drop_cols.append(colName)
            else:
                keep_cols.append(colName)

    for name in keep_cols: print(name)

    this_mlProfile.df.drop(columns=drop_cols, inplace=True)
    #make the target
    target_df = pd.DataFrame()
    y = []
    cut = 0.03
    for entry in this_mlProfile.df['MMM_7DayPctChange']:
        if entry > cut:
            y.append(1)
        elif  entry < -1*cut:
            y.append(0)
        else:
            y.append(-1)

    target_df['target'] = np.array(y)

    str_vals = [str(i) for i in y]
    print('Data spread y:', Counter(str_vals))

    """Split the data"""
    x_train, x_test, y_train, y_test = model_selection.train_test_split(this_mlProfile.df,
                target_df['target'],
                test_size = 0.3)

    """Declare our voting classifier and which classifiers it will use"""
    classifier_tuples = [('lsvc', svm.LinearSVC()),
                        ('nsvc', svm.NuSVC()),
                        ('knn', neighbors.KNeighborsClassifier()),
                        ('rfc', RandomForestClassifier(max_depth=5, n_estimators=50, random_state=1)),
                        ('gnb', GaussianNB()),
                        ('log', LogisticRegression(random_state=1)),
                        ('gpc', GaussianProcessClassifier(1.0* RBF(1.0))),
                        ('dtc', DecisionTreeClassifier(max_depth=5)),
                        ('mlp', MLPClassifier(alpha=1, max_iter=1000)),
                        ('ada', AdaBoostClassifier()),
                        ('qda', QuadraticDiscriminantAnalysis())]

    clf = VotingClassifier(
            estimators=[
                        #('lsvc', svm.LinearSVC()),
                        #('nsvc', svm.NuSVC()),
                        #('knn', neighbors.KNeighborsClassifier()),
                        ('rfc', RandomForestClassifier(max_depth=5, n_estimators=50, random_state=1)),
                        ('gnb', GaussianNB()),
                        ('log', LogisticRegression(random_state=1)),
                        ('gpc', GaussianProcessClassifier(1.0* RBF(1.0))),
                        #('dtc', DecisionTreeClassifier(max_depth=5)),
                        ('mlp', MLPClassifier(alpha=1, max_iter=1000)),
                        ('ada', AdaBoostClassifier()),
                        ('qda', QuadraticDiscriminantAnalysis())
                        ],
            voting='hard')

    clf.fit(x_train, y_train)
    confidence = clf.score(x_test, y_test)
    print('Accuracy', confidence)
    predictions = clf.predict(x_test)

    print('{} Predicted spread:'.format(ticker), Counter(predictions))

    return confidence

machineLearn('MMM')
