import pandas as pd

from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier

classifier_training_data_filename = "classification_train.data"
classifier_testing_data_filename = "classification_test.test"

classifier_training_data = pd.read_csv(classifier_training_data_filename, header=None)
classifier_testing_data = pd.read_csv(classifier_testing_data_filename, header=None)

classifier_y_train = classifier_training_data[48]
classifier_x_train = classifier_training_data.drop([48], axis=1)
classifier_x_test = classifier_testing_data.drop([48], axis=1)

classifier = AdaBoostClassifier() # chosen as best classifier in validation stage
classifier.fit(classifier_x_train, classifier_y_train)
classifier_predictions = classifier.predict(classifier_x_test)

classifier_output_filename = "classification_predictions.out"
classifier_output_file = open(classifier_output_filename, "w")
for item in classifier_predictions:
    classifier_output_file.write(str(item) + "\n")


regressor_training_data_filename = "regression_train.data"
regressor_testing_data_filename = "regression_test.test"

regressor_training_data = pd.read_csv(regressor_training_data_filename, header=None)
regressor_testing_data = pd.read_csv(regressor_testing_data_filename, header=None)

regressor_y_train = regressor_training_data[21]
regressor_x_train = regressor_training_data.drop([21], axis=1)
regressor_x_test = regressor_testing_data.drop([21], axis=1)

regressor = RandomForestRegressor() # chosen as best regressor in validation stage
regressor.fit(regressor_x_train, regressor_y_train)
regressor_predictions = regressor.predict(regressor_x_test)

regressor_output_filename = "regression_predictions.out"
regressor_output_file = open(regressor_output_filename, "w")
for item in regressor_predictions:
    regressor_output_file.write(str(item) + "\n")
