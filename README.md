# Time Series Forecasting Stock Prices based on historical price data and sentiment associated with company news


## To run this code, the data files are given for the stock data dataset. The link is already included in the code, so there should be no need to input it yourself for it to run.

## Please only run the code for the neural network (not the data preprocessing - that if for viewing but the output from that is already a part of the neural network code) 

## The following libraries are necessary to run the code (in code form)

### For Data Processing

(#pip install regex==2022.3.2)

(#pip install textblob)

(#pip install dateparser==1.1.1)

import pandas as pd

from pygooglenews import GoogleNews

import dateparser

import time

from time import mktime

from textblob import TextBlob

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer

from nltk import FreqDist

from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')

nltk.download('punkt')

### For Neural Network

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from scipy import stats


## Works Cited: 

“Where Developers Learn, Share, &amp; Build Careers.” Stack Overflow, stackoverflow.com/. 

  - Used this for a lot of troubleshooting

“Neural Networks from Scratch - p.1 Intro and Neuron Code.” YouTube, YouTube, 11 Apr. 2020, www.youtube.com/watch?v=Wo5dMEP_BbI&amp;list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3. 

  - Used this to help gain the foundation for starting this project

GitHub, github.com/fanghao6666/neural-networks-and-deep-learning/blob/master/py/Planar%20data%20classification%20with%20one%20hidden%20layer%20v3.py. 

  - Used this as a foundation of how to solve forward and backward propagation
