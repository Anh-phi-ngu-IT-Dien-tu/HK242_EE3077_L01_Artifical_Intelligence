from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, TimeDistributed, RepeatVector
import tensorflow as tf
import numpy as np

names = ['Long', 'Binh', 'Dung']
chars = sorted(set(''.join(names)))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(chars)
max_len = max(len(name) for name in names)

model = load_model("LSTM.h5")
model.summary()
def predict_name(char):
    idx = tf.constant([char2idx[char]], dtype=tf.int32)
    x = tf.one_hot(idx, depth=vocab_size)
    x = tf.expand_dims(x, axis=1)
    pred = model.predict(x, verbose=0)
    pred_indices = tf.argmax(pred[0], axis=1).numpy()
    return ''.join(idx2char[i] for i in pred_indices)

# --- Kết quả ---
for c in ['L', 'B', 'D']:
    print(f"Input: {c} → Output: {predict_name(c)}")