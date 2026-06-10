# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from Bio import SeqIO
import re
import numpy as np
import itertools
import math


# Superfamily dict
superf_dict = {
    'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
    'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
    'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
    'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
    'KOLOBOK': 28, 'ACADEM-1': 29
    }

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

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


def load_data(TE_lib, mode="T"):
    seqs = []
    classifications = []

    for te in SeqIO.parse(TE_lib, "fasta"):
        seqs.append(re.sub(r'[^ACGT]', '', str(te.seq).upper()))
        if mode == "T":
            superfamily = te.id.split(" ")[0].split("#")[1]
            classifications.append(superf_dict[superfamily])
        elif mode == "P":
            classifications.append(te.id)
        else:
            return None, None

    X = generate_mats(seqs)
    return np.asarray(X), np.asarray(classifications)


def word_seq(seq, k, stride=1):
    i = 0
    words_list = []
    while i <= len(seq) - k:
        words_list.append(seq[i: i + k])
        i += stride
    return (words_list)


def generate_kmer_dic (repeat_num):
    kmer_dic = {}

    bases = ['A','G','C','T']
    kmer_list = list(itertools.product(bases, repeat=int(repeat_num)))
    for eachitem in kmer_list:
        #print(eachitem)
        each_kmer = ''.join(eachitem)
        kmer_dic[each_kmer] = 0

    return (kmer_dic)


def generate_mat (words_list,kmer_dic):
    for eachword in words_list:
        kmer_dic[eachword] += 1

    num_list = []
    for eachkmer in kmer_dic:
        num_list.append(kmer_dic[eachkmer])

    return (num_list)


def generate_mats (seqs):
    seq_mats = []
    for eachseq in seqs:
        words_list = word_seq(eachseq, 7, stride=1)
        kmer_dic = generate_kmer_dic(7)
        num_list = generate_mat(words_list,kmer_dic)

        seq_mats.append(num_list)

    return seq_mats


def get_model(num_classes):
    model = Sequential()

    model.add(Conv2D(100, (1, 3), activation='relu', input_shape=(1, 16384, 1)))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(150, (1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(225, (1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(int(num_classes), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_m])
    return model


def run_experiment(model, X_train, Y_train, labels, X_dev, Y_dev, batch_size, num_epochs):
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_m', mode="max", factor=0.5, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_f1_m', mode="max", patience=15, restore_best_weights=True)

    X_train = X_train.astype("float32")
    Y_train = Y_train.astype("float32")
    X_dev = X_dev.astype("float32")
    Y_dev = Y_dev.astype("float32")

    batch_size = min(batch_size, X_train.shape[0], X_dev.shape[0])
    train_steps = math.ceil(X_train.shape[0] / batch_size)
    val_steps = math.ceil(X_dev.shape[0] / batch_size)
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, Y_train))
                .shuffle(min(len(X_train), 10000), reshuffle_each_iteration=True)
                .batch(batch_size, drop_remainder=False)
                .repeat()
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((X_dev, Y_dev))
              .batch(batch_size, drop_remainder=False)
              .repeat()
              .prefetch(tf.data.AUTOTUNE))

    history = model.fit(
        train_ds,
        epochs=num_epochs,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=[lr_scheduler, early_stopping],
        verbose=1
    )

    del train_ds
    del val_ds

    return history