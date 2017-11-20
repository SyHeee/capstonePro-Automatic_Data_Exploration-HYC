import GPy, GPyOpt
import numpy as np
import pandas as pds
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

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


bounds = [
          {'name': 'l1_drop', 'type': 'continuous', 'domain': (0.2, 0.6)},#0
          {'name': 'l2_drop', 'type': 'continuous', 'domain': (0.4, 0.8)},#1
          {'name': 'l3_drop', 'type': 'continuous', 'domain': (0.3, 0.7)},#2
          {'name': 'l4_drop', 'type': 'continuous', 'domain': (0.4, 0.8)},#3
          {'name': 'l1_out', 'type': 'discrete', 'domain': (200, 400, 600)},#4
          {'name': 'l2_out', 'type': 'discrete', 'domain': (80, 140,200)},#5
          {'name': 'l3_out', 'type': 'discrete', 'domain': (32 , 64, 128)},#6
          {'name': 'l4_out', 'type': 'discrete', 'domain': (16,26,40)},#7
          {'name': 'batch_size', 'type': 'discrete', 'domain': (16,32,64,128)}#8
          ]


# Neural Network

def f(x):
    print("\nSetting up neural network model...")
    early_stopping = EarlyStopping(patience=10, verbose=2)
    nn = Sequential()
    nn.add(Dense(units=int(x[:, 4]), kernel_initializer='normal', input_dim=x_allData.shape[1]))
    nn.add(PReLU())
    nn.add(Dropout(float(x[:, 0])))
    nn.add(Dense(units=int(x[:, 5]), kernel_initializer='normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(float(x[:, 1])))
    nn.add(Dense(units=int(x[:, 6]), kernel_initializer='normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(float(x[:, 2])))
    nn.add(Dense(units=int(x[:, 7]), kernel_initializer='normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(float(x[:, 3])))
    nn.add(Dense(1, kernel_initializer='normal'))
    nn.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))
    print("\nFitting neural network model...")
    nn.fit(x_training, y_training, batch_size=int(x[:, 8]), epochs=70, verbose=2, validation_split=0.1, callbacks=[early_stopping])
    loss_and_metrics=nn.evaluate(x_testing, y_testing, batch_size=int(x[:, 8]))
    print("\nEvaluating neural network model...")
    print("loss_and_metrics:",loss_and_metrics)
    return loss_and_metrics

# optimizer
opt_mnist = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)

# #### Running optimization

# In[8]:

# optimize mnist model
opt_mnist.run_optimization(max_iter=100)

# #### The output

# In[15]:

# print optimized mnist model
print("optimized parameters: {0}".format(opt_mnist.x_opt))
print("optimized loss: {0}".format(opt_mnist.fx_opt))