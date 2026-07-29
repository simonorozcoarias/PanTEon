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

def fasta2one_hot(sequence, total_win_len):
	langu = ['A', 'C', 'G', 'T', 'N']
	posNucl = 0
	if len(sequence) < total_win_len:
		rest = ['N' for x in range(total_win_len - len(sequence))]
		sequence += ''.join(rest)

	rep2d = np.zeros((1, 5, len(sequence)), dtype=np.int8)

	for nucl in sequence:
		posLang = langu.index(nucl.upper())
		rep2d[0][posLang][posNucl] = 1
		posNucl += 1
	return rep2d


def one_hot2fasta(dataset):
	langu = ['A', 'C', 'G', 'T', 'N']
	fasta_seqs = ""
	for j in range(dataset.shape[1]):
		if sum(dataset[:, j]) > 0:
			pos = argmax(dataset[:, j])
			fasta_seqs += langu[pos]
	return fasta_seqs


def r2_score(y_true, y_pred):
	# Sum of squared residuals
	sum_squares_residuals = tf.reduce_sum(tf.square(y_true - y_pred))

	# Mean of the true values
	mean_y_true = tf.reduce_mean(y_true)

	# Total sum of squares
	sum_squares = tf.reduce_sum(tf.square(y_true - mean_y_true))

	# Avoid division by zero
	epsilon = tf.keras.backend.epsilon()

	sum_squares = tf.maximum(sum_squares, epsilon)

	# Coefficient of determination (R2)
	R2 = 1 - sum_squares_residuals / sum_squares
	return R2


def load_data(fasta_path, max_len, inference=False):
	sequences = list(SeqIO.parse(fasta_path, "fasta"))

	if not inference:
		X = np.zeros((len(sequences), 5, max_len), dtype=np.int8)
		Y = np.zeros((len(sequences), 2), dtype=np.float16)

		for i, sequence in enumerate(tqdm(sequences, desc="Converting sequences to one-hot representation... ")):
			X[i, :, :] = fasta2one_hot(sequence.seq, max_len)
			Y[i, 0] = float(sequence.id.split(" ")[0].split("#")[0].split("_")[-2])
			Y[i, 1] = float(sequence.id.split(" ")[0].split("#")[0].split("_")[-1])

		return X, Y
	elif inference:
		X = np.zeros((len(sequences), 5, max_len), dtype=np.int8)
		labels_TEs = []
		for i, sequence in enumerate(tqdm(sequences, desc="Converting sequences to one-hot representation... ")):
			labels_TEs.append(sequence.id)
			X[i, :, :] = fasta2one_hot(sequence.seq, max_len)
		return X, labels_TEs
	else:
		return None, None


def get_model(shape1, shape2):
	# Inputs
	inputs = tf.keras.Input(shape=(shape1, shape2, 1), name="input_1")

	# layer 1
	layers = tf.keras.layers.Conv2D(32, (5, 51), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(0.01),
									kernel_regularizer=tf.keras.regularizers.l1_l2(0.0000, 0.000),
									bias_regularizer=tf.keras.regularizers.l1_l2(0.0000, 0.0000), use_bias=True)(inputs)
	layers = tf.keras.layers.SpatialDropout2D(0.2)(layers)
	layers = tf.keras.layers.AveragePooling2D((1, 9), strides=None)(layers)
	layers = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.6, epsilon=0.001, scale=False)(layers)

	# layer 2
	layers = tf.keras.layers.Conv2D(64, (1, 31), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(0.01),
									kernel_regularizer=tf.keras.regularizers.l1_l2(0.0000, 0.000),
									bias_regularizer=tf.keras.regularizers.l1_l2(0.0000, 0.0000), use_bias=True)(layers)
	layers = tf.keras.layers.SpatialDropout2D(0.2)(layers)
	layers = tf.keras.layers.AveragePooling2D((1, 9), strides=None)(layers)
	layers = tf.keras.layers.BatchNormalization(axis=1, momentum=0.6, epsilon=0.001, scale=False)(layers)

	# layer 3
	layers = tf.keras.layers.Conv2D(128, (1, 11), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(0.01),
									kernel_regularizer=tf.keras.regularizers.l1_l2(0.0000, 0.000),
									bias_regularizer=tf.keras.regularizers.l1_l2(0.0000, 0.0000), use_bias=True)(layers)
	layers = tf.keras.layers.SpatialDropout2D(0.2)(layers)
	layers = tf.keras.layers.AveragePooling2D((1, 7), strides=None)(layers)
	layers = tf.keras.layers.BatchNormalization(axis=1, momentum=0.6, epsilon=0.001, scale=False)(layers)

	# layer 4
	layers = tf.keras.layers.Conv2D(256, (1, 5), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(0.01),
									kernel_regularizer=tf.keras.regularizers.l1_l2(0.0000, 0.000),
									bias_regularizer=tf.keras.regularizers.l1_l2(0.0000, 0.0000), use_bias=True)(layers)
	layers = tf.keras.layers.SpatialDropout2D(0.2)(layers)
	layers = tf.keras.layers.AveragePooling2D((1, 5), strides=None)(layers)
	layers = tf.keras.layers.BatchNormalization(axis=1, momentum=0.6, epsilon=0.001, scale=False)(layers)

	# layer 5
	layers = tf.keras.layers.Flatten()(layers)

	# layer 6
	layers = tf.keras.layers.Dense(300, activation=tf.keras.layers.LeakyReLU(0.01),
								   kernel_regularizer=tf.keras.regularizers.l1_l2(0.0003, 0.001),
								   bias_regularizer=tf.keras.regularizers.l1(0.001))(layers)
	layers = tf.keras.layers.Dropout(0.2)(layers)
	layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, scale=False)(layers)

	# layer 7
	layers = tf.keras.layers.Dense(300, activation=tf.keras.layers.LeakyReLU(0.01),
								   kernel_regularizer=tf.keras.regularizers.l1_l2(0.0003, 0.001),
								   bias_regularizer=tf.keras.regularizers.l1(0.001))(layers)
	layers = tf.keras.layers.Dropout(0.2)(layers)
	layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, scale=False)(layers)

	# layer 8
	layers = tf.keras.layers.Dense(300, activation=tf.keras.layers.LeakyReLU(0.01),
								   kernel_regularizer=tf.keras.regularizers.l1_l2(0.0003, 0.001),
								   bias_regularizer=tf.keras.regularizers.l1(0.001))(layers)
	layers = tf.keras.layers.Dropout(0.2)(layers)
	layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, scale=False)(layers)

	# layer end
	predictions = tf.keras.layers.Dense(2, activation="sigmoid", name="output_1")(layers)

	# model generation
	model = tf.keras.Model(inputs=inputs, outputs=predictions)

	# optimizer
	opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	# loss function
	loss_fn = tf.keras.losses.MeanSquaredError()

	# Compile model
	model.compile(loss=loss_fn, optimizer=opt, metrics=[r2_score])
	return model


def run_experiment(model, X_train, Y_train, X_dev, Y_dev, batch_size, num_epochs):
	lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_r2_score', mode="max", factor=0.01, patience=10, verbose=1)
	early_stopping = EarlyStopping(monitor='val_r2_score', mode="max", patience=50, restore_best_weights=True)

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

