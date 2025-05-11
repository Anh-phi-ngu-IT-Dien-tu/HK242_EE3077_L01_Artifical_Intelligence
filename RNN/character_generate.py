import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, RepeatVector, TimeDistributed
import json

DEBBUGGING=True

names = ['Long', 'Binh', 'Dung']
chars = sorted(set(''.join(names)))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(chars) 
max_len = 3

input_indices = tf.constant([char2idx[name[0]] for name in names], dtype=tf.int32)
X = tf.one_hot(input_indices, depth=vocab_size)  # shape: (3, vocab)
X = tf.expand_dims(X, axis=1)  # shape: (3, 1, vocab)

# Đầu ra: chuỗi → one-hot từng ký tự
Y_indices = np.zeros((len(names), max_len), dtype=np.int32)
for i, name in enumerate(names):
    temp=''.join([name[i] for i in range(1,len(name))])

    for t, char in enumerate(temp):
        Y_indices[i, t] = char2idx[char]

Y = tf.one_hot(Y_indices, depth=vocab_size)  # shape: (3, max_len, vocab)
print(X.shape)
print(Y.shape)

if DEBBUGGING:
    model = Sequential([
        SimpleRNN(50, input_shape=(1, vocab_size), activation='tanh'),#hidden state la vector 50*1
        RepeatVector(max_len),
        SimpleRNN(50,return_sequences=True,activation='tanh'),#
        TimeDistributed(Dense(vocab_size, activation='softmax'))
    ])

    callback=tf.keras.callbacks.ModelCheckpoint(
        filepath="RNN.h5",
        monitor='loss', 
        patience=5, 
        restore_best_weights=True)

    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy'])

    # --- Huấn luyện ---
    history = model.fit(X, Y, epochs=500 ,callbacks=[callback] ,verbose=2)
    with open("history.json","w") as output:
        json.dump(history.history, output)
    model.summary()
