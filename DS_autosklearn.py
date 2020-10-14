import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import autosklearn.regression
import sklearn.metrics

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Data loading
file = "^GSPC.csv"
df = pd.read_csv(file, header=None, index_col=None, delimiter=',')
df['Adj_Close'] = df[5]

# Predict 'n' days out into the future
future_days = 1
df['Prediction'] = df[['Adj_Close']].shift(-future_days)

# Add the 5 day moving average technical indicator
df['MA_5'] = ta.MA(df[5].values, timeperiod=5, matype=0)
# Add the 20 day moving average technical indicator
df['MA_20'] = ta.MA(df[5].values, timeperiod=20, matype=0) 
# Add the 50 day moving average technical indicator
df['MA_50'] = ta.MA(df[5].values, timeperiod=50, matype=0)

stock_real = list(np.array(df[5]))
t_real = list(np.array(df[0]))
df.dropna(inplace=True)

# Create our target and labels
features_to_fit = ['MA_5', 'MA_20', 'MA_50', 'Adj_Close']
X = df[features_to_fit]
Y = df['Prediction']

# Split the data into 70% training and 30% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

automl = autosklearn.regression.AutoSklearnRegressor(per_run_time_limit=30, 
    metric=autosklearn.metrics.mean_squared_error)
automl.fit(X_train, Y_train)
print(automl.show_models())
print(automl.sprint_statistics())

Y_train_pred = automl.predict(X_train)
Y_test_pred = automl.predict(X_test)

# Quantify the quality of predictions
print("Mean Squared Error of train prediction:", 
    sklearn.metrics.mean_squared_error(Y_train, Y_train_pred))
print("Mean Squared Error of test prediction:", 
    sklearn.metrics.mean_squared_error(Y_test, Y_test_pred))
print("R2 Score of train prediction:", 
    sklearn.metrics.r2_score(Y_train, Y_train_pred))
print("R2 Score of test prediction:",
    sklearn.metrics.r2_score(Y_test, Y_test_pred))

# Plot
test_pred = list(Y_test_pred)
t_test = t_real[-len(test_pred):]

train_pred = list(Y_train_pred)
t_train = t_real[-(len(train_pred) + len(test_pred)):-len(test_pred)]

plt.plot(t_real, stock_real, 'r', t_train, train_pred, 'g', t_test, test_pred, 'b')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.show()