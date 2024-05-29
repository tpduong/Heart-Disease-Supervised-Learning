from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import graphviz
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve, GridSearchCV
from learningCurve import plot_learning_curve
from sklearn.svm import SVC


# import data
dataHeart = pd.read_csv('datasets/heart.csv')
dataWine = pd.read_csv('datasets/winequality-red.csv')

def support_vector_machine(data, name, tuning=False):
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
        param_grid = [
            {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
             'C': [1, 10, 100, 1000]},
            {'kernel': ['linear'],
             'C': [1, 10, 100, 1000]}]
        clf = GridSearchCV(SVC(), param_grid, cv=cv, scoring='accuracy')

        title = "SVM with tuning - " + name
    else:
        print("not tuning")
        clf = SVC(C=10, kernel='linear')
        title = "SMV with tuning - " + name

    clf.fit(X_train, y_train)

    if tuning:
        print("Best parameters set found on development set:", clf.best_params_)

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    print("here")
    plot_learning_curve(clf, title, X_train, y_train, axes=axes, ylim=(0.2, 1.01), cv=cv, n_jobs=4)

    print(clf.score(X_test, y_test))
    plt.show()


#support_vector_machine(dataHeart, "Heart Disease", tuning=False)
support_vector_machine(dataWine, "Red Wine Quality", tuning=False)

#support_vector_machine(dataHeart, "Heart Disease", tuning=True)
#support_vector_machine(dataWine, "Red Wine Quality", tuning=True)