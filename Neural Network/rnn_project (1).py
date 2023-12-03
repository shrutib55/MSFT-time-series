

import pandas as pd

stock_data = pd.read_csv("https://raw.githubusercontent.com/shrutib55/MSFT-time-series/main/MSFT-2.csv")


stock_data


stock_data.head()


pip install setuptools==58.0.0


pip install pygooglenews --upgrade


pip install feedparser==6.0.0


from pygooglenews import GoogleNews

gn = GoogleNews()


# pip install regex==2022.3.2
# pip install textblob

# pip install dateparser==1.1.1

import dateparser

start = dateparser.parse('11/06/2012')
end = dateparser.parse('11/06/2023')


search = gn.search('MSFT', helper = True, when='10y')


items = search['entries']


unfiltered_news = pd.DataFrame(items)


x = unfiltered_news['published_parsed'].tolist()
x


import time
from time import mktime

unfiltered_news['published_parsed'] = [datetime.fromtimestamp(mktime(y)).date() for y in x]




unfiltered_news



unfiltered_news.loc[:, ['title', 'published_parsed']]




from textblob import TextBlob





import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')
nltk.download('punkt')




def preprocess_text(text):
    
    text = text.lower()
    
    text = text.rsplit('-', 1)[-2]

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)



unfiltered_news['processed_title'] = unfiltered_news['title'].apply(preprocess_text)



unfiltered_news.loc[:, ['title', 'processed_title', 'published_parsed']]



def get_sentiment_polarity(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity



unfiltered_news['sentiment_polarity'] = unfiltered_news['processed_title'].apply(get_sentiment_polarity)


sent_data = unfiltered_news.loc[:, ['title', 'processed_title', 'sentiment_polarity', 'published_parsed']].sort_values(by='published_parsed')



pd.set_option('display.max_colwidth', None)




sent_data.sort_values(by='sentiment_polarity')



sent_data.rename(columns={"published_parsed": "Date"}, inplace = True)



print(stock_data.columns)



print(sent_data.columns)



stock_data['Date'] = pd.to_datetime(stock_data['Date'])
sent_data['Date'] = pd.to_datetime(sent_data['Date'])

total_data = pd.merge(stock_data, sent_data, on="Date", how='left')



total_data.to_csv('total_data.csv')



total_data.sort_values(by='Date').sort_values(by='sentiment_polarity').head(20)



total_data.dropna().sort_values(by='Date').head(30)



stock_data.dtypes



sent_data.dtypes



total_data.isna().sum()



unfiltered_news['summary'][0]







# ERRORS

# gn.search('MSFT', helper = True, from_='2022-11-06', to_='2023-11-06')

# dateparser.parse(unfiltered_news['published_parsed'])

# from datetime import datetime
# datetime(unfiltered_news['published_parsed'])



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



Works Cited: 

“Where Developers Learn, Share, &amp; Build Careers.” Stack Overflow, stackoverflow.com/. 


“Neural Networks from Scratch - p.1 Intro and Neuron Code.” YouTube, YouTube, 11 Apr. 2020, www.youtube.com/watch?v=Wo5dMEP_BbI&amp;list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3. 

GitHub, github.com/fanghao6666/neural-networks-and-deep-learning/blob/master/py/Planar%20data%20classification%20with%20one%20hidden%20layer%20v3.py. 
