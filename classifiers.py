import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

training_data_filename = "classification_train.data"
data = pd.read_csv(training_data_filename, header = None)

y = data[48]
x = data.drop([48], axis=1)

skf = StratifiedKFold(n_splits=10)

for train_index, test_index in skf.split(x, y):
    x_train = data.loc[train_index, list(range(48))]
    x_test = data.loc[test_index, list(range(48))]
    y_train = data.loc[train_index, [48]]
    y_test = data.loc[test_index, [48]]

    classifier1 = MLPClassifier()
    classifier1.fit(x_train, y_train)
    y_pred1 = classifier1.predict(x_test)

    classifier2 = DecisionTreeClassifier()
    classifier2.fit(x_train, y_train)
    y_pred2 = classifier2.predict(x_test)

    c1_accuracy_score = accuracy_score(y_pred1, y_test)
    c2_accuracy_score = accuracy_score(y_pred2, y_test)

    print "MLP Accuracy: {}\t Decision Tree Accuracy: {}".format(c1_accuracy_score, c2_accuracy_score)