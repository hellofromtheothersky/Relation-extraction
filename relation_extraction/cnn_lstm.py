from process_data import RE_DataEncoder
from BaseModel import BaseModel

import numpy as np
import seaborn as sns
import pickle

from keras.layers import Dense, Embedding, Conv1D

from tensorflow.keras.layers import (
    Embedding,
    Dense,
    Dropout,
    Input,
    concatenate,
    Reshape,
    LSTM,
)
from tensorflow.keras.layers import GlobalMaxPool1D
from keras.models import Model


def gen_glove_vector():
    with open("glove.6B.300d.txt", "r", encoding="UTF-8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
    emb_matrix = np.zeros(
        (Encoder.word_size + 1, 300)
    )  # vì word_index nó không quan tâm đến num_words đã set, nên
    for word, index in Encoder.word_index.items():
        embedding_vector = word_to_vec_map.get(word)
        if embedding_vector is not None:
            emb_matrix[index, :] = embedding_vector
    return emb_matrix


class CNN_LSTM_model(BaseModel):
    def build_model(self, using=["word_emb", "position_emb", "gram_emb", "sp_emb"]):
        input_sentence = Input(shape=(max_len,), name="sentence")

        embed_sentence_glove = Embedding(
            input_dim=Encoder.word_size + 1,
            output_dim=300,  # cần tương thích vs tham số weights bên dưới
            input_length=max_len,
            weights=[emb_matrix],
            name="sentence_glove",
            mask_zero=True,
        )(input_sentence)

        input_e1_pos = Input(shape=(max_len,), name="e1_position")
        embed_e1_pos = Embedding(126, 200, input_length=max_len, mask_zero=True)(
            input_e1_pos
        )
        input_e2_pos = Input(shape=(max_len,), name="e2_position")
        embed_e2_pos = Embedding(122, 200, input_length=max_len, mask_zero=True)(
            input_e2_pos
        )

        input_grammar = Input(shape=(max_len,), name="grammar_relation")
        embed_grammar = Embedding(45, 100, input_length=max_len, mask_zero=True)(
            input_grammar
        )

        input_sp = Input(shape=(max_len,), name="shortest_path")
        embed_sp = Embedding(62, 500, input_length=max_len, mask_zero=True)(input_sp)

        input_list = []
        if "word_emb" in using:
            input_list.append(embed_sentence_glove)
        if "position_emb" in using:
            input_list.extend([embed_e1_pos, embed_e2_pos])
        if "gram_emb" in using:
            input_list.append(embed_grammar)
        if "sp_emb" in using:
            input_list.append(embed_sp)

        visible = concatenate(input_list)
        interp = Conv1D(filters=306, kernel_size=5, activation="relu")(visible)
        interp = GlobalMaxPool1D()(interp)
        interp = Reshape((1, 306))(interp)
        interp = LSTM(122, dropout=0.4)(interp)
        output = Dense(19, activation="softmax")(interp)
        self.model = Model(
            inputs=[
                input_sentence,
                input_e1_pos,
                input_e2_pos,
                input_grammar,
                input_sp,
            ],
            outputs=output,
        )


with open("data/data_encoder.obj", "rb") as f:
    Encoder = pickle.load(f)

vocab_size = Encoder.vocab_size
max_len = Encoder.max_len
emb_matrix = gen_glove_vector()

X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

# CNN
cnn_lstm_model = CNN_LSTM_model()
cnn_lstm_model.build_model()
cnn_lstm_model.train_model(X_train, y_train, epochs=5)
cnn_lstm_model.evaluate(X_test, y_test, Encoder.dict_labels)
cnn_lstm_model.save_model("cnn_lstm")

# cnn_lstm_model=CNN_LSTM_model()
# cnn_lstm_model.load_model('cnn_lstm')
# cnn_lstm_model.evaluate(X_test, y_test, Encoder.dict_labels)
