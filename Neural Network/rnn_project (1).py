# -*- coding: utf-8 -*-
"""RNN_project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HewvcidDpU9Seh6XzXYDhi_2zxHOV93f
"""

import pandas as pd
import numpy as np

total_data = pd.read_csv("https://raw.githubusercontent.com/shrutib55/MSFT-time-series/main/Neural%20Network/total_data.csv", index_col = None)
total_data = total_data.loc[:,["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "title", "processed_title", "sentiment_polarity"]]
new_cols = ["Date", "Close", "Adj Close", "Open", "High", "Low", "Volume", "title", "processed_title", "sentiment_polarity"]
total_data = total_data[new_cols]

total_data.head(5)

#target:adj close
X=total_data.drop(['Close','title','processed_title', "Date"],axis=1)
#y=total_data['Adj Close']

#['Date', 'Close', 'Adj Close', 'Open', 'High', 'Low', 'Volume', 'title', 'processed_title', 'sentiment_polarity']
total_data.columns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

X['sentiment_polarity']=X['sentiment_polarity'].fillna(0)

X

#Dealing with outliers
from scipy import stats
z_scores=np.abs(stats.zscore(X[['Open','High','Low','Volume','sentiment_polarity','Adj Close']]))

X=X[(z_scores<3).all(axis=1)]

X2=total_data.drop(['Close','Adj Close','title','processed_title', 'Date'],axis=1)
removed=X2[~X2.index.isin(X.index)]

#Normalization
scaler=MinMaxScaler()
X[[ 'Adj Close', 'Open', 'High', 'Low', 'Volume']]=scaler.fit_transform(X[[ 'Adj Close', 'Open', 'High', 'Low', 'Volume']])

X

N=5
for i in range(1,N+1):
    X[f'Adj_Close_Lag_{i}']=X['Adj Close'].shift(i)

X

X.dropna(inplace=True)

X



#X=X.drop(['Date'],axis=1)

train_size=int(len(X)*0.8)
train,test=X[0:train_size],X[train_size:len(X)]

train.shape, test.shape

y_train=train['Adj Close'].values
train=train.drop(['Adj Close'],axis=1)
train=train.values
X_train=train.reshape((train.shape[0],1,train.shape[1]))

X_train.shape, y_train.shape

y_test=test['Adj Close'].values
test=test.drop(['Adj Close'],axis=1)
test=test.values
X_test=test.reshape((test.shape[0],1,test.shape[1]))

X_test.shape, y_test.shape

class RNN:
  def __init__(self, n_input, n_hidden, n_output):
    #initialize weights
    self.Wxh = np.random.randn(n_hidden, n_input) * 0.01
    self.Whh = np.random.randn (n_hidden, n_hidden) * 0.01
    self.Why = np.random.randn (n_output, n_hidden) * 0.01

    #initialize biases
    self.bh = np.zeros ((n_hidden, 1))
    self.by = np.zeros ((n_output, 1))

  def forward(self, X):
    h = np.zeros((self.Wxh.shape[0], 1))
    y_pred = []
    for t in range(X.shape[0]):
        x = X[t].reshape(-1, 1)
        h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
        y = np.dot(self.Why, h) + self.by
        y_pred.append(y)
    return np.array(y_pred).reshape(-1, 1), h

  def compute_loss(self, y_pred, y_true):
    return (np.array(y_pred - y_true.reshape(-1, 1))** 2).mean()

  def backward(self, X, y_true, y_pred, h_prev):
    #derivatives
    dWxh = np.zeros_like(self.Wxh)
    dWhh =  np.zeros_like(self.Whh)
    dWhy = np.zeros_like(self.Why)
    dbh  = np.zeros_like(self.bh)
    dby  = np.zeros_like(self.by)
    dh_next = np.zeros((self.Wxh.shape[0], 1))

    for t in reversed(range(X.shape[0])):
        dy = y_pred[t] - y_true  # derivative of loss with respect to y_pred, y_true is now a scalar
        dy = dy.reshape(-1, 1)  # Ensure dy is a column vector

        # Gradients for Why and by
        dWhy += np.dot(dy, h_prev[t].T)
        dby += dy

        # Backpropagate through time
        dh = np.dot(self.Why.T, dy) + dh_next
        dh_raw = (1 - h_prev[t] ** 2) * dh  # derivative of tanh

        x = X[t].reshape(-1, 1)  # Ensure X[t] is a column vector

        # Gradients for Wxh, Whh, and bh
        dWxh += np.dot(dh_raw, x.T)
        if t != 0:
            dWhh += np.dot(dh_raw, h_prev[t-1].T)
        dbh += dh_raw

        dh_next = np.dot(self.Whh.T, dh_raw)
        # Clip gradients for stability
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

    return dWxh, dWhh, dWhy, dbh, dby

  def train(self, X_train, y_train, epochs, learning_rate):
    for epoch in range(epochs+1):
      total_loss = 0
      for i in range(X_train.shape[0]):
        X, y_true = X_train[i], y_train[i]
        y_pred, h = self.forward(X)
        total_loss += self.compute_loss(y_pred, y_true)

        dWxh, dWhh, dWhy, dbh, dby = self.backward(X, y_true, y_pred, h)

        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

      avg_loss = total_loss / X_train.shape[0]
      if epoch % 10 == 0:
          print(f'Epoch {epoch}, Average Loss: {avg_loss}')

  def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred

  def plot_predictions_test(self, X_test, y_test):
      y_pred, _ = self.forward(X_test)

      plt.plot(y_test, label='True')
      plt.plot(y_pred, label='Predicted')

      plt.title('MSFT Stock Forecast')
      plt.xlabel('Days')
      plt.ylabel('Adjusted Close Price')
      plt.legend()
      plt.show()

#Start training
n_input = 10
n_hidden = 9
n_output = 1
rnn = RNN (n_input, n_hidden, n_output)

rnn.train(X_train, y_train, epochs=100, learning_rate = 0.001)

# Make predictions on the testing set
y_pred_test = rnn.predict(X_test)
loss_test = rnn.compute_loss(y_pred_test, y_test)
print(f'Average Loss: {loss_test}')

rnn.plot_predictions_test(X_test, y_test)

X1 = X.drop(['sentiment_polarity'],axis=1)
train_size2=int(len(X1)*0.8)
train1,test1=X1[0:train_size2],X1[train_size2:len(X)]

train1.shape, test1.shape
y_train1=train1['Adj Close'].values
train1=train1.drop(['Adj Close'],axis=1)

train1=train1.values
X_train1=train1.reshape((train1.shape[0],1,train1.shape[1]))

X_train1.shape, y_train1.shape


y_test1=test1['Adj Close'].values
test1=test1.drop(['Adj Close'],axis=1)
test1=test1.values
X_test1=test1.reshape((test1.shape[0],1,test1.shape[1]))

X_test1.shape, y_test1.shape

n_input = 9
n_hidden = 15
n_output = 1
rnn = RNN (n_input, n_hidden, n_output)
rnn.train(X_train1, y_train1, epochs= 1000, learning_rate = 0.01)
y_pred_test = rnn.predict(X_test1)
loss_test = rnn.compute_loss(y_pred_test, y_test1)
print(f'Average Loss: {loss_test}')
rnn.plot_predictions_test(X_test1, y_test1)