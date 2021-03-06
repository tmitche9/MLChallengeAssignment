import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

training_data_filename = "regression_train.data"
data = pd.read_csv(training_data_filename, header=None)

y = data[21]
x = data.drop([21], axis=1)

regressors = list()
regressors.append(LinearRegression())
regressors.append(AdaBoostRegressor())
regressors.append(RandomForestRegressor())
regressors.append(KNeighborsRegressor())

skf = StratifiedKFold(n_splits=10)
fold = 1

for train_index, test_index in skf.split(x, y):
    print "Fold {}\n-------".format(fold)
    fold = fold + 1

    x_train = data.loc[train_index, list(range(21))]
    x_test = data.loc[test_index, list(range(21))]
    y_train = data.loc[train_index, [21]]
    y_test = data.loc[test_index, [21]]

    for regressor in regressors:
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        mse = mean_squared_error(y_pred, y_test)
        print "{} MSE: {}".format(regressor.__class__.__name__, mse)

    print ""
