 #coding=utf8  
import nltk  
import numpy as np  
from keras import Input  
from numpy import array  
from numpy import asarray  
from numpy import zeros  
import tensorflow  
from keras.preprocessing.text import Tokenizer  
from keras.preprocessing.sequence import pad_sequences  
from keras import layers
from keras import models
from keras.layers import Embedding
import pandas as pds
from keras.preprocessing.sequence import pad_sequences
import gensim
import re
from collections import Counter
from keras.models import load_model
import  numpy as np
from keras.utils import to_categorical
from numpy import array
import pickle
import nltk  
from keras.layers.recurrent import LSTM
from keras.layers import Activation, Dense
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D  
#!pip install keras==2.2.5  
#nltk.download('punkt')  
!pip install pyspellChecker 

embedding_matrix = pickle.load(open('myWords.p', 'rb'))  


def create_model(vocabulary_size, seq_len):
    """
    model = models.Sequential()
    model.add(Embedding(vocabulary_size,
                        seq_len))
    model.add(SpatialDropout1D(0.3))
    model.add(Bidirectional(LSTM(seq_len*7, dropout=0.5, recurrent_dropout=0.5)))
    
    model.add(Dense(seq_len*5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(seq_len, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4,activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    """
    model =  models.Sequential()
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
    model.add(LSTM(400,input_shape=(1117, 100),return_sequences=True))
    model.add(LSTM(300))
    model.add(Dense(200, activation='relu'))

    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()

    return model  
    
    




