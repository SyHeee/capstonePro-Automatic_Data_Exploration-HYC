
import GPy, GPyOpt
import numpy as np
import pandas as pds
import random
from keras.layers import Activation, Dropout, BatchNormalization, Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

x_allData = np.loadtxt("../intermediate/x_allData.txt")
y_allData = np.loadtxt("../intermediate/y_allData.txt")
x_training, x_testing, y_training, y_testing = train_test_split(x_allData, y_allData, test_size=0.1, random_state=42)
print("x_training.shape:",x_training.shape)
print("x_testing.shape:",x_testing.shape)
print("y_training.shape:",y_training.shape)
print("y_testing.shape:",y_testing.shape)
print("x_allData.shape:",x_allData.shape)
print("y_allData.shape:",y_allData.shape)

class MNIST():
    def __init__(self, first_input=x_allData.shape[1], last_output=1,
                 l1_out=512,
                 l2_out=512,
                 l1_drop=0.2,
                 l2_drop=0.2,
                 batch_size=100,
                 epochs=50,
                 validation_split=0.1):
        self.__first_input = first_input
        self.__last_output = last_output
        self.l1_out = l1_out
        self.l2_out = l2_out
        self.l1_drop = l1_drop
        self.l2_drop = l2_drop
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.__x_train = x_training
        self.__x_test=x_testing
        self.__y_train=y_training
        self.__y_test=y_testing
        self.__model = self.mnist_model()


    # mnist model
    def mnist_model(self):
        model = Sequential()
        model.add(Dense(self.l1_out, input_shape=(self.__first_input,)))
        model.add(Activation('relu'))
        model.add(Dropout(self.l1_drop))
        model.add(Dense(self.l2_out))
        model.add(Activation('relu'))
        model.add(Dropout(self.l2_drop))
        model.add(Dense(self.__last_output))
        # model.add(Activation('softmax'))
        model.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))

        return model

    # fit mnist model
    def mnist_fit(self):
        early_stopping = EarlyStopping(patience=10, verbose=2)

        self.__model.fit(self.__x_train, self.__y_train,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         verbose=2,
                         validation_split=self.validation_split,
                         callbacks=[early_stopping])

    # evaluate mnist model
    def mnist_evaluate(self):
        self.mnist_fit()

        evaluation = self.__model.evaluate(self.__x_test, self.__y_test, batch_size=self.batch_size, verbose=2)
        return evaluation


def run_mnist(first_input=x_allData.shape[1], last_output=1,
              l1_out=512, l2_out=512,
              l1_drop=0.2, l2_drop=0.2,
              batch_size=100, epochs=50, validation_split=0.1):
    _mnist = MNIST()
    mnist_evaluation = _mnist.mnist_evaluate()
    return mnist_evaluation


bounds = [{'name': 'validation_split', 'type': 'continuous', 'domain': (0.0, 0.3)},
          {'name': 'l1_drop', 'type': 'continuous', 'domain': (0.0, 0.3)},
          {'name': 'l2_drop', 'type': 'continuous', 'domain': (0.0, 0.3)},
          {'name': 'l1_out', 'type': 'discrete', 'domain': (64, 128, 256, 512, 1024)},
          {'name': 'l2_out', 'type': 'discrete', 'domain': (64, 128, 256, 512, 1024)},
          {'name': 'batch_size', 'type': 'discrete', 'domain': (10, 100, 500)},
          {'name': 'epochs', 'type': 'discrete', 'domain': (20, 60)}]


# #### Bayesian Optimization

# In[6]:

# function to optimize mnist model
def f(x):
    print("Parameters Setting: ", x)
    evaluation = run_mnist(
        l1_drop=int(x[:, 1]),
        l2_drop=int(x[:, 2]),
        l1_out=float(x[:, 3]),
        l2_out=float(x[:, 4]),
        batch_size=int(x[:, 5]),
        epochs=int(x[:, 6]),
        validation_split=float(x[:, 0]))
    # print("loss:{0} \t\t accuracy:{1}".format(evaluation[0], evaluation[1]))
    print("Test MAE: ", evaluation)
    return evaluation


# #### Optimizer instance

# In[7]:

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


# In[ ]: