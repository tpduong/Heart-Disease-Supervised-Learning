from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import graphviz
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve, GridSearchCV
from learningCurve import plot_learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# import data
dataHeart = pd.read_csv('datasets/heart.csv')
dataWine = pd.read_csv('datasets/winequality-red.csv')

def knn(data, name, tuning=False):
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
        print("tunnninnnngggggg....")
        k = np.arange(3, 20)

        param_grid = [
            {
               'n_neighbors': k,
               'weights': ['uniform', 'distance'],
               'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        ]
        clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring='accuracy')

        title = "kNN with tuning - " + name
    else:
        print("not tuning")
        clf = KNeighborsClassifier()
        title = "kNN without tuning - " + name

    clf.fit(X_train, y_train)

    if tuning:
        print("Best parameters set found on development set:", clf.best_params_)

    # fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    # plot_learning_curve(clf, title, X_train, y_train, axes=axes, ylim=(0.2, 1.01), cv=cv, n_jobs=4)
    #
    # print(clf.score(X_test, y_test))
    # plt.show()


#sknn(dataHeart, "Heart Disease", tuning=False)
#support_vector_machine(dataWine, "Red Wine Quality", tuning=False)

knn(dataHeart, "Heart Disease", tuning=True)
#knn(dataWine, "Red Wine Quality", tuning=True)