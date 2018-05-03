import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

training_data_filename = "regression_train.data"
data = pd.read_csv(training_data_filename, header=None)

y = data[21]
x = data.drop([21], axis=1)

skf = StratifiedKFold(n_splits=10)

for train_index, test_index in skf.split(x, y):
    x_train = data.loc[train_index, list(range(21))]
    x_test = data.loc[test_index, list(range(21))]
    y_train = data.loc[train_index, [21]]
    y_test = data.loc[test_index, [21]]

    regressor1 = LinearRegression()
    regressor1.fit(x_train, y_train)
    y_pred1 = regressor1.predict(x_test)

    regressor2 = AdaBoostRegressor()
    regressor2.fit(x_train, y_train)
    y_pred2 = regressor2.predict(x_test)

    r1_mse = mean_squared_error(y_pred1, y_test)
    r2_mse = mean_squared_error(y_pred2, y_test)

    print "Linear Regressor MSE: {}\t AdaBoost Regressor MSE: {}".format(r1_mse, r2_mse)