from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

import pandas as pd
import numpy as np
import graphviz
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve, GridSearchCV
from learningCurve import plot_learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC




# import data
dataHeart = pd.read_csv('datasets/heart.csv')
dataWine = pd.read_csv('datasets/winequality-red.csv')

def boosting(data, name, tuning=False):
    # split data into train and test set
    trainData, testData = train_test_split(data, test_size=0.2)

    X_train = trainData.iloc[:, :-1]
    y_train = trainData.iloc[:, -1:].values.ravel()

    X_test = testData.iloc[:, :-1]
    y_test = testData.iloc[:, -1:].values.ravel()

    # map wine quality to good/bad >5 good, <=5 bad
    if name == "Red Wine Quality":
        y_train = np.array(list(map(lambda x: 1 if x > 5 else 0, y_train)))
        y_test = np.array(list(map(lambda x: 1 if x > 5 else 0, y_test)))


    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    if tuning:
        if name == "Heart Disease":
            dTree = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=7, min_samples_split=5,
                                           splitter='random')
        if name == "Red Wine Quality":
            dTree = DecisionTreeClassifier(ccp_alpha=0.005, criterion='entropy', max_depth=8, min_samples_split=2,
                                           splitter='random')
        mySvm = SVC(probability=True , kernel='linear')

        param_grid = [
            {
               'base_estimator': [dTree, mySvm],
               'n_estimators' : [50, 100, 200, 250],
            }
        ]
        clf = GridSearchCV(AdaBoostClassifier(), param_grid, cv=cv, scoring='accuracy')

        title = "Adaboost with tuning - " + name
    else:
        if name == "Heart Disease":
            dTree = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=7, min_samples_split=5,
                                       splitter='random')
        if name == "Red Wine Quality":
            dTree = DecisionTreeClassifier(ccp_alpha=0.005, criterion='entropy', max_depth=8, min_samples_split=2,
                                           splitter='random')

        clf = AdaBoostClassifier(base_estimator=dTree, n_estimators=250)
        title = "Adaboost with tuning - " + name

    clf.fit(X_train, y_train)

    if tuning:
        print("Best parameters set found on development set:", clf.best_params_)

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    plot_learning_curve(clf, title, X_train, y_train, axes=axes, ylim=(0.2, 1.01), cv=cv, n_jobs=4)

    print(clf.score(X_test, y_test))
    plt.show()


boosting(dataHeart, "Heart Disease", tuning=False)
#boosting(dataWine, "Red Wine Quality", tuning=False)

#boosting(dataHeart, "Heart Disease", tuning=True)
#boosting(dataWine, "Red Wine Quality", tuning=True)