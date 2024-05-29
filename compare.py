from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import graphviz
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from learningCurve import plot_learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC



# import data
dataHeart = pd.read_csv('datasets/heart.csv')
dataWine = pd.read_csv('datasets/winequality-red.csv')

def test(clf, name, X_train, y_train, X_test, y_test):
    result={}
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    result["Train"] = end - start

    start = time.time()
    accuracy = clf.score(X_test, y_test)
    end = time.time()
    result["Query"] = end - start
    result["Accuracy"] = accuracy

    return result


def compare(data, name):
    # split data into train and test set
    trainData, testData = train_test_split(data, test_size=0.2)

    X_train = trainData.iloc[:, :-1]
    y_train = trainData.iloc[:, -1:].values.ravel()

    X_test = testData.iloc[:, :-1]
    y_test = testData.iloc[:, -1:].values.ravel()


    if name == "Heart Disease":
        dTree = DecisionTreeClassifier(ccp_alpha=0.01, criterion='gini', max_depth=7, min_samples_split=5,
                                       splitter='random')
        neuralNetwork = MLPClassifier(activation='identity', hidden_layer_sizes=(18,), solver='lbfgs', max_iter=200)
        boosting = AdaBoostClassifier(base_estimator=dTree, n_estimators=250)
        svm = SVC(C=1, kernel='linear')
        kNN = KNeighborsClassifier(algorithm='auto', n_neighbors=10, weights='uniform')

     # map wine quality to good/bad >5 good, <=5 bad
    if name == "Red Wine Quality":
        y_train = np.array(list(map(lambda x: 1 if x > 5 else 0, y_train)))
        y_test = np.array(list(map(lambda x: 1 if x > 5 else 0, y_test)))

        # models
        dTree = DecisionTreeClassifier(ccp_alpha=0.005, criterion='entropy', max_depth=8, min_samples_split=2,
                                           splitter='random')
        neuralNetwork = MLPClassifier(activation='relu', hidden_layer_sizes=(10,), solver='lbfgs', max_iter=250)
        boosting = AdaBoostClassifier(base_estimator=dTree, n_estimators=250)
        svm = SVC(C=10, kernel='linear')
        kNN = KNeighborsClassifier(algorithm='auto', n_neighbors=19, weights='distance')


    result = test(dTree, "dTree", X_train, y_train, X_test, y_test)
    print("dTree", result)

    result = test(neuralNetwork, "neuralNetwork", X_train, y_train, X_test, y_test)
    print("neuralNetwork", result)

    result = test(boosting, "boosting", X_train, y_train, X_test, y_test)
    print("boosting", result)

    result = test(svm, "svm", X_train, y_train, X_test, y_test)
    print("svm", result)

    result = test(kNN, "kNN", X_train, y_train, X_test, y_test)
    print("kNN", result)

#compare(dataHeart, "Heart Disease")
compare(dataWine, "Red Wine Quality")