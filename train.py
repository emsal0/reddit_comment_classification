#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# maybe these depend on the reddit data...

top_words = 5000
output_dim = 64
input_dim = 600
num_lstm = 256
num_labels = 256

num_epochs = 4
batch_size = 64

model = Sequential()

model.add(Embedding(top_words, output_dim, input_length=input_dim))
model.add(Dropout(0.2))
model.add(LSTM(num_lstm))
model.add(Dropout(0.2))
model.add(Dense(num_labels, activation='sigmoid'))
# model.fit(X_train, y_train, nb_epoch=num_epochs, batch_size=batch_size)
