# -*- coding: utf-8 -*-
"""LY_relation_extraction_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Z7Hczjha7h8NPH12xTGyZHQ65LWBQeSo
"""

# from google.colab import drive
# drive.mount('/content/drive/')

# !cp /content/drive/MyDrive/nlp/relation_extraction/process_data.py /content

"""# import"""

# BASE_DIR="/content/drive/MyDrive/nlp/relation_extraction/"
BASE_DIR=""

import process_data
import importlib
importlib.reload(process_data)
from process_data import RE_DataEncoder
from process_data import get_feature


import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

from keras.layers import Dense, Embedding, Conv1D
from keras import Sequential

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.layers import Embedding, Dense, Dropout, Input, concatenate
from tensorflow.keras.layers import GlobalMaxPool1D
from keras.models import Model
from keras.utils import plot_model

with open(BASE_DIR+'/data/data_encoder.obj', 'rb') as f:
    Encoder=pickle.load(f)

vocab_size=Encoder.vocab_size
max_len= Encoder.max_len

def evaluate(y_test, preds):
    print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), zero_division=0, target_names=list(Encoder.lbencoder.classes_)))
    print(Encoder.dict_labels)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 7))
    sns.heatmap(confusion_matrix(y_test.argmax(axis=1), preds.argmax(axis=1)), annot=True, fmt=".1f")


def save_model(model, model_name):
    path=BASE_DIR+'/saved_model/'
    model_structure=model.to_json()
    with open(path+model_name+'.json', 'w') as json_file:
        json_file.write(model_structure)
    model.save_weights(path+model_name+'.h5')


def load_model(model_name):
    path=BASE_DIR+'/saved_model/'
    from keras.models import model_from_json
    with open(path+model_name+'.json', "r") as rf:
        jstr=rf.read()
    model=model_from_json(jstr)
    model.load_weights(path+model_name+'.h5')
    return model

  
def read_glove_vector(glove_vec):
  with open(glove_vec, 'r', encoding='UTF-8') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
      w_line = line.split()
      curr_word = w_line[0]
      word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
  return word_to_vec_map


def gen_glove_vector():
    word_to_vec_glove = read_glove_vector(BASE_DIR+'/glove.6B.300d.txt')

    emb_matrix = np.zeros((Encoder.word_size+1, 300)) # vì word_index nó không quan tâm đến num_words đã set, nên 
    for word, index in Encoder.word_index.items():
        embedding_vector = word_to_vec_glove.get(word)
        if embedding_vector is not None:
            emb_matrix[index, :] = embedding_vector
    return emb_matrix

"""# Main

## prepare data and layer
"""

X_train = np.load(BASE_DIR+'/data/X_train.npy')
X_test = np.load(BASE_DIR+'/data/X_test.npy')
y_train = np.load(BASE_DIR+'/data/y_train.npy')
y_test = np.load(BASE_DIR+'/data/y_test.npy')

emb_matrix=gen_glove_vector()

# np.unique(X_train[4].flatten())

input_sentence = Input(shape=(max_len,), name='sentence')
embed_sentence = Embedding(input_dim=vocab_size, output_dim=300, input_length=max_len, mask_zero=True)(input_sentence)

embed_sentence_glove = Embedding(input_dim=Encoder.word_size+1, 
                            output_dim=300, #cần tương thích vs tham số weights bên dưới
                            input_length=max_len,
                            weights = [emb_matrix],
                            name='sentence_glove', mask_zero=True)(input_sentence)

input_e1_pos = Input(shape=(max_len,), name='e1_position')
embed_e1_pos = Embedding(126,200, input_length=max_len, mask_zero=True)(input_e1_pos)
input_e2_pos = Input(shape=(max_len,), name='e2_position')
embed_e2_pos = Embedding(122,200, input_length=max_len, mask_zero=True)(input_e2_pos)

input_grammar = Input(shape=(max_len,), name='grammar_relation')
embed_grammar = Embedding(45,100, input_length=max_len, mask_zero=True)(input_grammar)

input_sp = Input(shape=(max_len,), name='shortest_path')
embed_sp = Embedding(62, 500, input_length=max_len, mask_zero=True)(input_sp)

"""## Define CNN model

"""

def CNN_model(input_list):
    visible = concatenate([input for input in input_list])
    interp = Conv1D(filters=100, kernel_size=3, activation='relu')(visible)
    interp = GlobalMaxPool1D()(interp)
    interp = Dropout(0.2)(interp)
    output = Dense(19, activation='softmax')(interp)
    model = Model(inputs=[input_sentence, input_e1_pos, input_e2_pos, input_grammar, input_sp], outputs=output)
    return model

model=CNN_model([embed_sentence_glove
    ,embed_e1_pos
    ,embed_e2_pos
    # , embed_grammar
    ,embed_sp
    ])
print(model.summary())

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

history = model.fit([X_train[0], X_train[1], X_train[2], X_train[3], X_train[4]], 
                    y_train, 
                    epochs=5,
                    batch_size=32,
                    validation_split=0.1)

save_model(model, 'cnn_nonbert')

model=load_model('cnn_nonbert')

label_test_pred = model.predict([X_test[0], X_test[1], X_test[2], X_test[3], X_test[4]])
print(evaluate(y_test, label_test_pred))
