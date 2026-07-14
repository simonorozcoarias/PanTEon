# -*- coding: utf-8 -*-
import tensorflow as tf
from Bio import SeqIO
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tqdm import tqdm
import seaborn as sn
import joblib
from itertools import product
import math

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

# Superfamily dict
superf_dict = {
    'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
    'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
    'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
    'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
    'KOLOBOK': 28, 'ACADEM-1': 29
    }

def recall_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def load_data(fasta_path, mode="T"):
    sequences = list(SeqIO.parse(fasta_path, "fasta"))
    k_range = (1, 6)

    all_kmer_set = set()
    k_min, k_max = k_range
    a = tuple(("A", "C", "G", "T"))
    for k in range(k_min, k_max + 1):
        for tup in product(a, repeat=k):
            all_kmer_set.add(''.join(tup))
    all_kmers = sorted(all_kmer_set)
    kmer_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}

    if mode == "T":
        counts_matrix = np.zeros((len(sequences), len(all_kmers) + 1), dtype=int)
        for seq_idx, seq in enumerate(tqdm(sequences, desc="Counting k-mers")):
            classification = seq.id.split(" ")[0].split("#")[1]
            if classification in superf_dict:
                order = superf_dict[classification]
                counts_matrix[seq_idx, 0] = order
                seq_str = str(seq.seq)
                for k in range(k_range[0], k_range[1] + 1):
                    for i in range(len(seq_str) - k + 1):
                        kmer = seq_str[i:i + k].upper()
                        if kmer in kmer_index:
                            counts_matrix[seq_idx, kmer_index[kmer] + 1] += 1
            else:
                print(f"[ERROR] {classification} not found in our superfamily dictionary")

        Y = counts_matrix[:, 0]
        X = counts_matrix[:, 1:]

        return X, Y
    elif mode == "P":
        counts_matrix = np.zeros((len(sequences), len(all_kmers)), dtype=int)
        labels_TEs = []
        for seq_idx, seq in enumerate(tqdm(sequences, desc="Counting k-mer frequencies... ")):
            labels_TEs.append(seq.id)
            seq_str = str(seq.seq)
            for k in range(k_range[0], k_range[1] + 1):
                for i in range(len(seq_str) - k):
                    kmer = seq_str[i:i + k].upper()
                    if kmer in kmer_index:
                        counts_matrix[seq_idx, kmer_index[kmer]] += 1
        return counts_matrix, labels_TEs
    else:
        return None, None


def get_model(shape, num_classes):
    inputs = tf.keras.Input(shape=(shape,))
    x = tf.keras.layers.Dense(200, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(200, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(200, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[f1_m])
    return model


def run_experiment(model, X_train, Y_train, labels, X_dev, Y_dev, batch_size, num_epochs):
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_m', mode="max", factor=0.01, patience=10, verbose=1)
    early_stopping = EarlyStopping(monitor='val_f1_m', mode="max", patience=50, restore_best_weights=True)

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
