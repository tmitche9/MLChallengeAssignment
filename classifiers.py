import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

training_data_filename = "classification_train.data"
data = pd.read_csv(training_data_filename, header=None)

y = data[48]
x = data.drop([48], axis=1)

classifiers = list()
classifiers.append(MLPClassifier())
classifiers.append(DecisionTreeClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(AdaBoostClassifier())

skf = StratifiedKFold(n_splits=10)
fold = 1

for train_index, test_index in skf.split(x, y):
    print "Fold {}\n-------".format(fold)
    fold = fold + 1

    x_train = data.loc[train_index, list(range(48))]
    x_test = data.loc[test_index, list(range(48))]
    y_train = data.loc[train_index, [48]]
    y_test = data.loc[test_index, [48]]

    for classifier in classifiers:
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        print "{} Accuracy: {}".format(classifier.__class__.__name__, accuracy)

    print ""
