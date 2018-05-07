import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

training_data_filename = "regression_train.data"
data = pd.read_csv(training_data_filename, header=None)

y = data[21]
x = data.drop([21], axis=1)

regressors = list()
regressors.append(RandomForestRegressor(n_estimators=175))
regressors.append(RandomForestRegressor(warm_start=True))
regressors.append(RandomForestRegressor(n_estimators=175, warm_start=True))

skf = StratifiedKFold(n_splits=10)
fold = 1

scores = [0, 0, 0]
for train_index, test_index in skf.split(x, y):
	print "Fold {}\n-------".format(fold)
	fold = fold + 1

	x_train = data.loc[train_index, list(range(21))]
	x_test = data.loc[test_index, list(range(21))]
	y_train = data.loc[train_index, [21]]
	y_test = data.loc[test_index, [21]]

	for k, regressor in enumerate(regressors):
		regressor.fit(x_train, y_train)
		y_pred = regressor.predict(x_test)
		mse = mean_squared_error(y_pred, y_test)
		scores[k] += mse
		print "{} MSE: {}".format(regressor.__class__.__name__, mse)
	
	print ""

#get mean values
scores = [1.*score/10 for score in scores]

for score in scores:
	print "Overall MSE: {}".format(score)
