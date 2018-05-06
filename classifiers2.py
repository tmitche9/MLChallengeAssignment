import pandas as pd

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

training_data_filename = "classification_train.data"
data = pd.read_csv(training_data_filename, header=None)

y = data[48]
x = data.drop([48], axis=1)

classifiers = list()
classifiers.append(AdaBoostClassifier(n_estimators=100, learning_rate=1.2))
classifiers.append(AdaBoostClassifier(n_estimators=100, learning_rate=1.2))
classifiers.append(AdaBoostClassifier(n_estimators=100, learning_rate=1.2))

skf = StratifiedKFold(n_splits=10)
fold = 1

scores = [0, 0, 0]
for train_index, test_index in skf.split(x, y):
    print "Fold {}\n-------".format(fold)
    fold = fold + 1

    x_train = data.loc[train_index, list(range(48))]
    x_test = data.loc[test_index, list(range(48))]
    y_train = data.loc[train_index, [48]]
    y_test = data.loc[test_index, [48]]

    for k, classifier in enumerate(classifiers):
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        scores[k] += accuracy
        print "{} Accuracy: {}".format(classifier.__class__.__name__, accuracy)
    
    print ""

#get mean value
scores = [1.*score/10 for score in scores]

for score in scores:
    print "Overall Accuracy: {}".format(score)

