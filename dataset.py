# imports libraries
import pickle										# import/export lists
import datetime										# dates
import string										# string parsing
import re 											# regular expression
import pandas as pd									# dataframes
import numpy as np									# numerical computation
from run import train

###############################################################################

# opens cleaned data
with open ("C:/Users/bendo/OneDrive/Documents/Sentiment Analysis/Lily's Final/fanfictionstatistics/data/clean_data/df_story", 'rb') as fp:
    df = pickle.load(fp)
df = df.loc[(df.state == 'online') & (df.language == 'English'), ].copy()
df.index = range(df.shape[0])

###############################################################################


#Set Parameters
X_var = 'summary'
y_var = 'genre'
max_feature_length = 10
num_classes = 20
learning_rate = .1
batch_size = 10
num_epochs = 10
embedding_size = 5

train(df, max_feature_length, num_classes, embedding_size, learning_rate, batch_size, num_epochs)
