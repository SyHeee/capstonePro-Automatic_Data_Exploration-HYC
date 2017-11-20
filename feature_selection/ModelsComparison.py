import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


x_allData = np.loadtxt("../intermediate/x_allData.txt")
y_allData = np.loadtxt("../intermediate/y_allData.txt")
# x_toPredict = np.loadtxt("../intermediate/x_toPredict.txt")

x_training, x_testing, y_training, y_testing = train_test_split(x_allData, y_allData, test_size=0.1, random_state=42)
print("x_training.shape:",x_training.shape)
print("x_testing.shape:",x_testing.shape)
print("y_training.shape:",y_training.shape)
print("y_testing.shape:",y_testing.shape)
print("x_allData.shape:",x_allData.shape)
print("y_allData.shape:",y_allData.shape)

ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,
                                       random_state=0),
    "K-nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
    "Random Forest": RandomForestRegressor(max_depth=2, random_state=0)
}

for name, estimator in ESTIMATORS.items():
    estimator.fit(x_training, y_training)
    y_predict=estimator.predict(x_testing)
    loss_and_metrics = mean_absolute_error(y_testing, y_predict)
    print(name, " loss_and_metrics:", loss_and_metrics)