import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

class multiModelTest:

    def __init__(self, x_train, y_train, x_test, y_test):
        print("init")

    def testing(self, x_train, y_train, x_test, y_test):
    #""" Get the loss(mse or r^2?) of models that you want to test

    #Parameters
    #----------
    #x_train, x_test, are of shape=[numOfSamples, numOfFeatures]
    #y_train, y_test, are of shape=[numOfSamples]
    #return parameter loss is of shape=[numOfSamples, numOfModels]
    #"""
        trainer = linear_model.LinearRegression()
        trainer.fit(x_train, y_train)
        y_pred = trainer.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        loss = np.array([1.0, 2, 3, 4, 5, 6, 7, 8]);
        loss = mse
        return loss
