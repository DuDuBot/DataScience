import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv

from sklearn.preprocessing import MinMaxScaler
from keras.layers.core import Dense, Activation, Dropout
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def initialize_data(data, look_back=1):
    x, y = [], []
    for i in range(len(data) - look_back - 1):
        temp1 = data[i: (i + look_back), 0]
        x.append(temp1)
        temp2 = data[i + look_back, 0]
        y.append(temp2)
    return np.array(x), np.array(y)


file = "^GSPC.csv"
np.random.seed(5)
df = read_csv(file, header=None, index_col=None, delimiter=',')

# column 5 is adj_close
adj_close = df[5].values
data = adj_close.reshape(-1, 1)
mmscaler = MinMaxScaler(feature_range=(0, 1))
data = mmscaler.fit_transform(data)

train_size = int(len(data) * 0.5)
test_size = len(data) - train_size
train, test = data[0:train_size, :], data[train_size:len(data), :]

timestep = 240

train_X, train_Y = initialize_data(train, timestep)
test_X, test_Y = initialize_data(test, timestep)
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

model = Sequential()
model.add(LSTM(25, input_shape=(1, timestep)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(train_X, train_Y, epochs=95, batch_size=240, verbose=1)

train_pred = model.predict(train_X)
test_pred = model.predict(test_X)

train_pred = mmscaler.inverse_transform(train_pred)
train_Y = mmscaler.inverse_transform([train_Y])
test_pred = mmscaler.inverse_transform(test_pred)
test_Y = mmscaler.inverse_transform([test_Y])

# root mean squared error
train_score = math.sqrt(mean_squared_error(train_Y[0], train_pred[:, 0]))
print('Train Score: %.2f' % train_score)
test_score = math.sqrt(mean_squared_error(test_Y[0], test_pred[:, 0]))
print('Test Score: %.2f' % test_score)

train_plot = np.empty_like(data)
train_plot[:, :] = np.nan
train_plot[timestep: len(train_pred) + timestep, :] = train_pred
test_plot = np.empty_like(data)
test_plot[:, :] = np.nan
test_plot[len(train_pred) + (timestep * 2) + 1: len(data)-1, :] = test_pred

plt.plot(mmscaler.inverse_transform(data))
plt.plot(train_plot)
trans = data[test_size + timestep:]
test_price = mmscaler.inverse_transform(trans)

p = np.around(list(test_pred.reshape(-1)), decimals=2)
tp = np.around(list(test_price.reshape(-1)), decimals=2)
df = pd.DataFrame(data={"prediction": p, "test_price": tp})
df.to_csv("result.csv", sep=';', index=None)
plt.plot(test_plot)
plt.show()
