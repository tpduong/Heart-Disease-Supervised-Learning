import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter


# import hearts disease dataset
# data = pd.read_csv('datasets/heart.csv')

data = pd.read_csv('datasets/winequality-red.csv')

# # ----------------------------------------------------
# # Using KNN to find attribute with highest correlation
# #-----------------------------------------------------
# # run the algorithm this many times and find average
# runs = 10
#
# accuracy_N_C_K = []
# for run in range(runs):
#     #split training and test data 80:20
#     trainData, testData = train_test_split(data, test_size=0.2)
#
#     # look at all combinations of 2 columns to find highest correlation to reduce the curse of dimensionality
#     accuracy_C_K = []
#     accuracy_count = []
#     for column1 in range(data.shape[1]-1):
#         for column2 in range (data.shape[1]-1):
#             if column1 != column2:
#                 XTrain = trainData.iloc[:, [column1, column2]]
#                 yTrain = trainDatat["target"]
#                 XTest = testData.iloc[:, [column1, column2]]
#                 yTest = testData["target"]
#
#                 accuracyK = []
#                 for k in range(1, 100):
#                     clf = neighbors.KNeighborsClassifier(k, weights='uniform')
#                     clf.fit(XTrain, yTrain)
#
#                     yPred = clf.predict(XTest)
#
#                     accuracy = accuracy_score(yTest, yPred)
#                     if accuracy > .75:
#                         accuracy_count.append(column1)
#                         accuracy_count.append(column2)
#
#                     accuracyK.append(accuracy)
#
#                 accuracy_C_K.append(accuracyK)
#
#     #accuracy_N_C_K.append(accuracy_C_K)
#
# #finalAccuracy = np.array(accuracy_N_C_K)
# #averageAccuracy = np.mean(finalAccuracy, axis=0)
# #print(averageAccuracy.shape)
# #print(averageAccuracy)
#
# #print(accuracy)
# print(accuracy_count)
# print(Counter(accuracy_count))
# Best attributes are at index 2, 12, 9, and 8
# Counter({2: 1362, 12: 1346, 9: 1194, 8: 1117, 5: 686, 6: 628, 1: 620, 11: 512, 10: 379, 3: 90, 7: 90, 4: 34, 0: 18})
# ---------------------------------------------

# ---------------------------------------------
# Run Knn with attributes 2, 12, 9, and 8, look for best k

runs = 10
accuracies = {}
for k in range(1,100):
    accuracy_K = []
    for run in range(runs):
        # split training and test data 80:20
        trainData, testData = train_test_split(data, test_size=0.2)
        XTrain = trainData.iloc[:, :-1]
        yTrain = trainData.iloc[:, -1:]

        XTest = testData.iloc[:, :-1]
        yTest = testData.iloc[:, -1:]

        clf = neighbors.KNeighborsClassifier(k, weights='uniform')
        clf.fit(XTrain, yTrain.values.ravel())

        yPred = clf.predict(XTest)

        accuracy = accuracy_score(yTest, yPred)
        accuracy_K.append(accuracy)

    npAccuracyK = np.array(accuracy_K)
    mean = np.mean(npAccuracyK)

    accuracies[k] = mean
    #if mean > .8:
    print(k, mean)
