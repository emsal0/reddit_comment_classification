#!/usr/bin/env python3

import numpy as np
import csv
from keras.models import Input, Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence, Tokenizer, one_hot
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

reddit_comments_path = "data/reddit_comments_2017_08_reddit_comments_2017_08.csv"
embeddings_path = "embeddings/glove.6B.50d.txt"
model_path = "classification_model.h5"

comments = []
subreddits = [] # basically labels in this case

print("processing data...")
with open(reddit_comments_path, 'r', encoding='latin1') as f:
    reader = csv.DictReader(f)
    for comment in reader:
        comments.append(comment['body'])
        subreddits.append(comment['subreddit'])
print("data processed")

train_size = int((len(subreddits)/4.0)*3) # len(subreddits) * 3/4

# Tokenize the reddit stuff

print("tokenizing comments...")
max_features = 50000
word_tokenizer = Tokenizer(max_features)
word_tokenizer.fit_on_texts(comments)

comments_tf = word_tokenizer.texts_to_sequences(comments)
print("comments tokenized")

# One-hot the labels

print("one-hotting...")

labelenc = LabelEncoder()
integer_encoded = labelenc.fit_transform(subreddits)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehotenc = OneHotEncoder(sparse = True)
subreddits_one_hot = onehotenc.fit_transform(integer_encoded)

print("finished one-hotting")
print("subreddits_one_hot = ")
print(subreddits_one_hot)

X_train = comments_tf[:train_size]
y_train = subreddits_one_hot[:train_size]
X_test = comments_tf[train_size:]
y_test = subreddits_one_hot[train_size:]

output_dim = 50
input_dim = 600
num_lstm = 8
num_labels = labelenc.classes_.size

num_epochs = 30
batch_size = 64

embedding_vectors = {}

print("pretraining embedding layer")
with open(embeddings_path, 'r') as f:
    for line in f:
        line_split = line.strip().split(" ")
        vec = np.array(line_split[1:], dtype=float)
        word = line_split[0]
        embedding_vectors[word] = vec

weights_matrix = np.zeros((max_features + 1, 50))

for word, i in word_tokenizer.word_index.items():
    embedding_vector = embedding_vectors.get(word)
    if embedding_vector is not None and i < max_features:
        weights_matrix[i] = embedding_vector
print("embedding layer pretrained")

model = Sequential()
model.add(Embedding(max_features + 1, output_dim, weights=[weights_matrix]))
model.add(Dropout(0.2))
model.add(LSTM(num_lstm))
model.add(Dropout(0.2))
model.add(Dense(num_labels, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

batch_size = 32
epochs = 20

print("training...")
model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)

scores = model.evaluate(X_test, y_test, verbose=1)

print("Accuracy: %.2f%%" % scores[1] * 100)
model.save(model_path)
