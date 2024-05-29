import pandas as pd
import numpy as np
import graphviz
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve, GridSearchCV
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from learningCurve import plot_learning_curve

def plot_tree(clf, X, y):
    clf = clf.fit(X, y)
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("myTree")

# import data
dataHeart = pd.read_csv('datasets/heart.csv')
dataWine = pd.read_csv('datasets/winequality-red.csv')

# ------------------------------------------------
# Decision tree w/o hyperparameter tuning
# ------------------------------------------------
def dTreeWOTuning(data, name):
    trainData, testData = train_test_split(data, test_size=0.1)
    # split data into train and test set
    X = trainData.iloc[:, :-1]
    y = trainData.iloc[:, -1:].values.ravel()

    title = "Decision Tree w/o tuning - " + name

    #reduce dimensions of X
    # if name == "Heart Disease":
    #     X = SelectKBest(chi2, k=2).fit_transform(X, y)
    #     title = "Learning curve with reduced features - " + name

    # map wine quality to good/bad >5 good, <=5 bad
    if name == "Red Wine Quality":
        y = np.array(list(map(lambda x: 1 if x > 5 else 0, y)))


    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = tree.DecisionTreeClassifier()
    #print(estimator.tree_.node_count)
    plot_learning_curve(estimator, title, X, y, axes=axes, ylim=(0.2, 1.01), cv=cv, n_jobs=4)

    #plot_tree(estimator, X, y)

    plt.show()

# ------------------------------------------------
# Decision tree with hyperparameter tuning
# ------------------------------------------------
def dTreeWTuning(data, name):
    trainData, testData = train_test_split(data, test_size=0.2)
    # split data into train and test set
    X = trainData.iloc[:, :-1]
    y = trainData.iloc[:, -1:].values.ravel()

    #reduce dimensions of X
    if name == "Heart Disease":
        X = SelectKBest(chi2, k=4).fit_transform(X, y)

    # map wine quality to good/bad >5 good, <=5 bad
    if name == "Red Wine Quality":
        y = np.array(list(map(lambda x: 1 if x > 5 else 0, y)))

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    title = "Decision Tree with tuning - " + name
    estimator = tree.DecisionTreeClassifier()
    max_depths = np.arange(2, 10)
    ccp_alpha = np.arange(0, 10) * .005
    parameters = {'criterion':('gini', 'entropy'), 'splitter':('best', 'random'), 'min_samples_split': [2, 3, 4, 5],
                  'max_depth':max_depths, 'ccp_alpha': ccp_alpha }
    # parameters = {'criterion':('gini', 'entropy')}
    # parameters = {'splitter':('best', 'random')}
    # parameters = {'max_depth':max_depths}

    clf = GridSearchCV(estimator, parameters)
    clf.fit(X, y)

    # for param in clf.cv_results_:
    #     print(param, clf.cv_results_[param])

    i = np.where(clf.cv_results_['rank_test_score'] == 1)
    index = i[0][0]
    tunedParams = clf.cv_results_['params'][index]
    print("tunedParams", tunedParams)

    cv = ShuffleSplit(n_splits=100, test_size=0.1, random_state=0)
    estimator = tree.DecisionTreeClassifier(**tunedParams)
    #print(estimator.tree_.node_count)
    plot_learning_curve(estimator, title, X, y, axes=axes, ylim=(0.5, 1.01), cv=cv, n_jobs=4)

    #plt.show()
    #plot_tree(estimator, X, y)





dTreeWOTuning(dataHeart, "Heart Disease")
#dTreeWOTuning(dataWine, "Red Wine Quality")
#dTreeWTuning(dataHeart, "Heart Disease")
#dTreeWTuning(dataWine, "Red Wine Quality")


