# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
from Bio import SeqIO
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from sklearn.utils import shuffle
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, GRU, Lambda
from sklearn.preprocessing import OneHotEncoder
import math

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

# Superfamily dict
superf_dict = {'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
               'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
               'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
               'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
               'KOLOBOK': 28, 'ACADEM-1': 29}

# Add attention layer to the deep learning network
class attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1), initializer="random_normal",
                                 trainable=True)
        super(attention, self).build(input_shape)

    def call(self, x, **kwargs):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x, self.W))
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        return context, alpha


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def create_gru_model(len_thre):
    # Define GRU model architecture
    rnn_input = Input(shape=(len_thre, 4), name="rnn_input")
    x = GRU(128, dropout=0.2, return_sequences=True, name="rnn1")(rnn_input)
    x = GRU(64, dropout=0.2, return_sequences=True, name="rnn2")(x)
    x = Flatten(name="rnn_flatten")(x)
    x = Dropout(0.5, name="rnn_dropout1")(x)
    x = Dense(128, activation="relu", name="rnn_fc1")(x)
    model = Model(inputs=rnn_input, outputs=x, name="GRU_Model")
    return model


def create_cnn_model(k):
    cnn_input = Input(shape=(1, pow(4, k), 1), name="cnn_input")
    x = Conv2D(filters=64, kernel_size=(1, 3), activation="relu", name="cnn_conv1")(cnn_input)
    x = MaxPool2D(pool_size=(1, 2), name="cnn_pool1")(x)
    x = Conv2D(filters=128, kernel_size=(1, 3), activation="relu", name="cnn_conv2")(x)
    x = MaxPool2D(pool_size=(1, 2), name="cnn_pool2")(x)
    x = Conv2D(filters=256, kernel_size=(1, 3), activation="relu", name="cnn_conv3")(x)
    x = MaxPool2D(pool_size=(1, 2), name="cnn_pool3")(x)
    x = Flatten(name="cnn_flatten")(x)
    x = Dropout(0.5, name="cnn_dropout1")(x)
    x = Dense(128, activation="relu", name="cnn_fc1")(x)
    model = Model(inputs=cnn_input, outputs=x, name="CNN_Model")
    return model


def create_attn_model(kmer, len_thre, class_num):
    cnn_model = create_cnn_model(kmer)
    rnn_model = create_gru_model(len_thre)
    merge_layer = Lambda(lambda x: tf.stack(x, axis=1), name="stack_layer")(
        [cnn_model.output, rnn_model.output]
    )
    attn_layer, attention_weight = attention(name="attention")(merge_layer)
    attn_layer = Lambda(lambda x: K.sum(x, axis=1), name="attn_sum")(attn_layer)
    dense_layer = Dense(128, activation="relu", name="fc")(attn_layer)
    dropout_layer = Dropout(0.5, name="dropout")(dense_layer)
    output_layer = Dense(int(class_num), activation="softmax", name="output")(dropout_layer)
    model = Model(inputs=[cnn_model.input, rnn_model.input], outputs=output_layer)
    attention_model = Model([cnn_model.input, rnn_model.input], attention_weight)
    return model, attention_model


def run_experiment(model, X_oh_train, X_oh_test, X_kmer_train, X_kmer_test, y_train, y_test, class_num, k, l, batch_size, epochs):
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_m', mode="max", factor=0.01, patience=10, verbose=1)
    early_stopping = EarlyStopping(monitor='val_f1_m', mode="max", patience=50, restore_best_weights=True)

    X_oh_train = X_oh_train.astype("float32")
    X_kmer_train = X_kmer_train.astype("float32")
    y_train = y_train.astype("float32")
    X_oh_test = X_oh_test.astype("float32")
    X_kmer_test = X_kmer_test.astype("float32")
    y_test = y_test.astype("float32")

    X_oh_train, X_kmer_train, y_train = shuffle(X_oh_train, X_kmer_train, y_train)
    X_oh_test, X_kmer_test, y_test = shuffle(X_oh_test, X_kmer_test, y_test)

    batch_size = min(batch_size, X_oh_train.shape[0], X_oh_test.shape[0])
    train_steps = math.ceil(X_oh_train.shape[0] / batch_size)
    val_steps = math.ceil(X_oh_test.shape[0] / batch_size)

    train_ds = (tf.data.Dataset.from_tensor_slices(((X_kmer_train, X_oh_train), y_train))
                .shuffle(min(len(X_oh_train), 10000), reshuffle_each_iteration=True)
                .batch(batch_size, drop_remainder=False)
                .repeat()
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices(((X_kmer_test, X_oh_test), y_test))
              .batch(batch_size, drop_remainder=False)
              .repeat()
              .prefetch(tf.data.AUTOTUNE))

    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=[lr_scheduler, early_stopping],
        verbose=1
    )

    del train_ds
    del val_ds

    return history


def convert_undetermined_base(seq):
    seq = seq.upper()
    seq = seq.replace("Y", "C")
    seq = seq.replace("D", "G")
    seq = seq.replace("S", "C")
    seq = seq.replace("R", "G")
    seq = seq.replace("V", "A")
    seq = seq.replace("K", "G")
    seq = seq.replace("N", "T")
    seq = seq.replace("H", "A")
    seq = seq.replace("W", "A")
    seq = seq.replace("M", "C")
    seq = seq.replace("X", "G")
    seq = seq.replace("B", "C")
    return seq


def seq2oh(seq, len_thre):
    if len(seq) >= len_thre:
        seq_1 = list(seq)[0:len_thre//2]
        seq_2 = list(seq)[-len_thre//2:]
        seq = seq_1 + seq_2
    else:
        seq_1 = list(seq)[0:len(seq)//2]
        seq_2 = list(seq)[-len(seq)//2:]
        seq = seq_1 + [0] * (len_thre - len(seq)) + seq_2
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(np.array(["A", "C", "G", "T"]).reshape(-1, 1))
    seq_encode = enc.transform(np.array(seq).reshape(-1, 1)).toarray()
    return seq_encode


def nucle2num(nucleotide):
    nucleotide = nucleotide.upper()
    if nucleotide == "A":
        return 0
    elif nucleotide == "G":
        return 1
    elif nucleotide == "C":
        return 2
    elif nucleotide == "T":
        return 3
    else:
        raise ValueError(f"Invalid nucleotide: {nucleotide}")


def seq2kmer(seq, k):
    b = 4 # base
    h = 0 # hash value
    hash_dict = {key: 0 for key in range(4 ** k)}
    if len(seq) >= k:
        # Initialize hash value
        for i in range(k):
            h = h * b + nucle2num(seq[i])
        hash_dict[h] = 1
        # Calculate frequency of remaining k-mers
        for i in range(k, len(seq)):
            h = (h - nucle2num(seq[i-k]) * b**(k-1)) * b + nucle2num(seq[i])
            hash_dict[h] += 1
    else:
        raise ValueError(f"Invalid k-mer size: {k}")
    return hash_dict


def get_kmer_data(data_file, k):
    kmer = []
    for record in SeqIO.parse(data_file, "fasta"):
        seq = convert_undetermined_base(str(record.seq))
        kmer.append(list(seq2kmer(seq, k).values()))
    X_kmer = np.array(kmer)
    return X_kmer


def get_oh_data(data_file, len_thre):
    one_hot = []
    for record in SeqIO.parse(data_file, "fasta"):
        seq = convert_undetermined_base(str(record.seq))
        one_hot.append(seq2oh(seq, len_thre))
    X_oh = np.array(one_hot)
    return X_oh


def get_label_data(data_file, mode="T"):
    labels = []
    for record in SeqIO.parse(data_file, "fasta"):
        if mode == "T":
            classification = record.id.split(" ")[0].split("#")[1]
            labels.append(superf_dict[classification])
        elif mode == "P":
            labels.append(record.id)
        else:
            return None
    return np.asarray(labels)