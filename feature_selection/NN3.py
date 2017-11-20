import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

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


# Neural Network
print("\nSetting up neural network model...")
nn = Sequential()
nn.add(Dense(units=400, kernel_initializer='normal', input_dim=x_allData.shape[1]))
nn.add(PReLU())
nn.add(Dropout(.4))
nn.add(Dense(units=160, kernel_initializer='normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(units=120, kernel_initializer='normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units=80, kernel_initializer='normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units=64, kernel_initializer='normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units=26, kernel_initializer='normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(1, kernel_initializer='normal'))
nn.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))

print("\nFitting neural network model...")
nn.fit(x_training, y_training, batch_size=32, epochs=36, verbose=2)
loss_and_metrics=nn.evaluate(x_testing, y_testing, batch_size=32)
print("\nEvaluating neural network model...")
print("loss_and_metrics:",loss_and_metrics)

# print("\nPredicting with neural network model...")
# print("x_test.shape:",x_test.shape)
# y_pred_ann = nn.predict(x_test)

# print("\nPreparing results for write...")
# nn_pred = y_pred_ann.flatten()
# print("Type of nn_pred is ", type(nn_pred))
# print("Shape of nn_pred is ", nn_pred.shape)

# print("\nNeural Network predictions:")
# print(pd.DataFrame(nn_pred).head())