import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve, GridSearchCV
from learningCurve import plot_learning_curve
from sklearn.neural_network import MLPClassifier

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# import data
dataHeart = pd.read_csv('datasets/heart.csv')
dataWine = pd.read_csv('datasets/winequality-red.csv')

@ignore_warnings(category=ConvergenceWarning)
def makeModel(data, name):
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


    param_grid = [
        {
           'activation': ['identity', 'logistic', 'tanh', 'relu'],
           'solver' : ['lbfgs', 'sgd', 'adam'],
           'hidden_layer_sizes': [
                         (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,)
                         ],
            'max_iter': [250]
        }
    ]
    #clf = GridSearchCV(MLPClassifier(), param_grid, cv=cv, scoring='accuracy')

    # for param in clf.cv_results_:
    #     print(param, clf.cv_results_[param])

    clf = MLPClassifier()
    clf.fit(X_train, y_train)




    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    title = "Neural Network with tuning - " + name
    plot_learning_curve(clf, title, X_train, y_train, axes=axes, ylim=(0.2, 1.01), cv=cv, n_jobs=4)



    print(clf.score(X_train, y_train))



    #print("Best parameters set found on development set:", clf.best_params_)

    #plt.show()



#makeModel(dataHeart, "Heart Disease")
makeModel(dataWine, "Red Wine Quality")