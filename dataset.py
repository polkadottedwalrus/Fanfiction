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

train(input_file)
