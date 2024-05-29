from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

from sklearn import tree


iris = datasets.load_iris()
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
parameters = {}

svc = tree.DecisionTreeClassifier()
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)
print(iris.keys())
print(iris.feature_names)

resultKeys = sorted(clf.cv_results_.keys())
for key in resultKeys:
    print(key, clf.cv_results_[key]
)