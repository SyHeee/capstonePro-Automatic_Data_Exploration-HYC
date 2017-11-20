import numpy as np
from sklearn.model_selection import train_test_split
from MultiModelTest import multiModelTest

from sklearn import linear_model
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
#from feature_selection.MultiModelTest import multiModelTest

ESTIMATORS = {
    "Linear Regression": linear_model.LinearRegression(),
    "Lasso Regression": linear_model.Lasso(alpha=0.5),
    "Elastic Net": linear_model.ElasticNet(alpha=0.5, l1_ratio=0.7),
    "Ridge": linear_model.Ridge(fit_intercept=False),
    "Lasso Lars": linear_model.LassoLars(alpha=0.5),
    "Bayesian Ridge": linear_model.BayesianRidge(compute_score=True),
    "AdaBoost": ensemble.AdaBoostRegressor(),
    "Bagging": ensemble.BaggingRegressor(),
    "Extra trees": ensemble.ExtraTreesRegressor(n_estimators=10, max_features=32, random_state=0),
    "K -nn": KNeighborsRegressor(),
}

ESTIMATORS_SINGLE = {
    "Linear Regression": linear_model.LinearRegression(),
    "Lasso Regression": linear_model.Lasso(alpha=0.5),
    "Elastic Net": linear_model.ElasticNet(alpha=0.5, l1_ratio=0.7),
    "Ridge": linear_model.Ridge(fit_intercept=False),
    "Lasso Lars": linear_model.LassoLars(alpha=0.5),
    "Bayesian Ridge": linear_model.BayesianRidge(compute_score=True),
    "AdaBoost": ensemble.AdaBoostRegressor(),
    "Bagging": ensemble.BaggingRegressor(),
    "Extra trees": ensemble.ExtraTreesRegressor(n_estimators=10, max_features=1, random_state=0),
    "K -nn": KNeighborsRegressor(),
}

"""
Plz use Preprocessing.py to generate intermediate results first.
"""

x_allData = np.loadtxt("../intermediate/x_allData.txt")
y_allData = np.loadtxt("../intermediate/y_allData.txt")
# x_toPredict = np.loadtxt("../intermediate/x_toPredict.txt")

x_training, x_testing, y_training, y_testing = train_test_split(x_allData, y_allData, test_size=0.1, random_state=42)

try:
    file_read = open('../output/allDataLoss.txt')
except:
    pktnum = 0

allDataLoss = np.array([])
outfile = open('../output/allDataLoss.txt', 'w')
outfile.write("Start:\n")
outfile = open('../output/allDataLoss.txt', 'a')
#First we put all the data to all the models to have a look.
for name, estimator in ESTIMATORS.items():
    estimator.fit(x_training, y_training)
    y_pred = estimator.predict(x_testing)
    curLoss = mean_absolute_error(y_testing, y_pred)
    allDataLoss = np.append(name, curLoss)
    outfile.write(name)
    outfile.write(str(curLoss) + '\n')
    print(name, "loss_and_merics:", allDataLoss)

#np.savetxt("../output/allDataLoss.txt",allDataLoss)

#Then we try every single feature on the model
#singleDataLoss = np.array([])
#for col in range(0, x_allData.shape[1]):
#    singleDataLoss = np.append(singleDataLoss, col)
#    singleDataLoss = np.append(singleDataLoss,trainingProcg(x_training.reshape(-1,1), x_testing[:,col], y_training, y_testing))
#singleDataLoss=singleDataLoss.reshape(x_allData.shape[1], int(singleDataLoss.shape[0]/x_allData.shape[1]));
#np.savetxt("../output/singleDataLoss.txt",singleDataLoss)
try:
    file_read = open('../output/singleDataLoss.txt')
except:
    pktnum = 0

outf = open('../output/singleDataLoss.txt', 'w')
outf.write("Start:\n")
outf = open('../output/singleDataLoss.txt', 'a')
singleDataLoss = np.array([])
sigFeaPredict = []
sigFModel = "init"
for col in range(0, x_allData.shape[1]):
    tmpLoss = []
    for name, estimator in ESTIMATORS_SINGLE.items():
        #print(x_training[:,col].reshape(-1,1))
        estimator.fit(x_training[:,col].reshape(-1,1), y_training)
        y_pred = estimator.predict(x_testing[:,col].reshape(-1,1))
        curLoss = mean_absolute_error(y_testing, y_pred)
        tmpLoss.append(curLoss)
        print(name,  curLoss)
        if curLoss == min(tmpLoss):
            sigFeaPredict = y_pred
            sigFModel = name
    curmin = min(tmpLoss)
    outf.write("colum number " + str(col) + ": ")
    outf.write(sigFModel + "[" + str(curmin) + "]\n")

corrMatrix = np.corrcoef(x_allData.transpose())
threshold = np.percentile(np.abs(corrMatrix), 10)
pairDataLoss = np.array([])
for row in range(0, corrMatrix.shape[0]):
    for col in range(row+1, corrMatrix.shape[1]):
        if(np.abs(corrMatrix[row][col])>threshold):
            pairDataLoss=np.append(pairDataLoss, [row,col])
            pairDataLoss=np.append(pairDataLoss, multi.testing(x_training[:,col], x_testing[:,col], y_training, y_testing))
pairDataLoss.reshape(int(pairDataLoss.shape[0]/(allDataLoss.shape[0]+2)),(allDataLoss.shape[0]+2))
np.savetxt("../output/pairDataLoss.txt",pairDataLoss)
