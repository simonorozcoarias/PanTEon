# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, AutoModel, BertModel
from Bio import SeqIO
import numpy as np
import math
import torch
from torch.nn.utils.rnn import pad_sequence

ALPHABET = 'ACGT'
superf_dict = {'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
				   'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
				   'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
				   'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
				   'KOLOBOK': 28, 'ACADEM-1': 29}


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
_trans = np.full(256, 255, dtype=np.uint8)
_trans[ord('A')] = 0; _trans[ord('C')] = 1; _trans[ord('G')] = 2; _trans[ord('T')] = 3

class CustomTokenizer:
	def __init__(self, vocab_file, k):
		self.k = k
		self.vocab = []
		self.token2id = {}
		with open(vocab_file, 'r') as f:
			for idx, line in enumerate(f):
				token = line.strip()
				self.vocab.append(token)
				self.token2id[token] = idx

		# Special tokens
		self.pad_token = '[PAD]'
		self.unk_token = '[UNK]'
		self.cls_token = '[CLS]'
		self.sep_token = '[SEP]'

		self.pad_token_id = self.token2id[self.pad_token]
		self.unk_token_id = self.token2id[self.unk_token]
		self.cls_token_id = self.token2id[self.cls_token]
		self.sep_token_id = self.token2id[self.sep_token]

	def tokenize(self, sequence):
		tokens = []
		for i in range(len(sequence) - self.k + 1):
			kmer = sequence[i:i + self.k]
			if kmer in self.token2id:
				tokens.append(kmer)
			else:
				tokens.append(self.unk_token)
		return tokens

	def convert_tokens_to_ids(self, tokens):
		return [self.token2id.get(token, self.unk_token_id) for token in tokens]

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


def load_data(TE_lib, mode="T", stride=1, max_tokens=503, bert_model_name="tools/bert-mini", vocab_file="data/kmer_vocab.txt"):
	seqs_1k, seqs_1250, seqs_1500, labels = [], [], [], []

	for te in SeqIO.parse(TE_lib, "fasta"):
		seq = str(te.seq).upper()
		if mode == "T":
			superf = te.id.split(" ")[0].split("#")[1]
			labels.append(superf_dict[superf])
		elif mode == "P":
			labels.append(te.id)
		else:
			return None, None, None, None

		seqs_1500.append(truncate_seq_indv(seq, 1500))
		seqs_1250.append(truncate_seq_indv(seq, 1250))
		seqs_1k.append(truncate_seq_indv(seq, 1000))

	labels = np.asarray(labels)

	X4_counts = np.vstack([kmer_counts_numpy(s, 4, stride) for s in seqs_1k])
	X5_counts = np.vstack([kmer_counts_numpy(s, 5, stride) for s in seqs_1250])
	X6_counts = np.vstack([kmer_counts_numpy(s, 6, stride) for s in seqs_1500])

	X4_counts = normalize_rows(X4_counts.astype(np.float32))
	X5_counts = normalize_rows(X5_counts.astype(np.float32))
	X6_counts = normalize_rows(X6_counts.astype(np.float32))

	model = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)

	tokenizer = CustomTokenizer(vocab_file, 4)
	X4_cls = bert_cls_embeddings_batched(seqs_1k,    model, tokenizer, k=4,  batch_size=256, max_tokens=max_tokens)
	tokenizer = CustomTokenizer(vocab_file, 5)
	X5_cls = bert_cls_embeddings_batched(seqs_1250,  model, tokenizer, k=5,  batch_size=256, max_tokens=max_tokens)
	tokenizer = CustomTokenizer(vocab_file, 6)
	X6_cls = bert_cls_embeddings_batched(seqs_1500,  model, tokenizer, k=6,  batch_size=256, max_tokens=max_tokens)

	X4 = np.concatenate([X4_cls, X4_counts], axis=1)
	X5 = np.concatenate([X5_cls, X5_counts], axis=1)
	X6 = np.concatenate([X6_cls, X6_counts], axis=1)

	return X4.astype(np.float32), X5.astype(np.float32), X6.astype(np.float32), labels


def kmer_counts_numpy(seq: str, k: int, stride: int = 1) -> np.ndarray:
	a = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
	d = _trans[a]                     # 0..3 o 255 (inválido)
	if len(d) < k:
		return np.zeros(4**k, dtype=np.uint32)

	w = np.lib.stride_tricks.sliding_window_view(d, k)[::stride]
	valid = np.all(w != 255, axis=1)
	w = w[valid]
	if w.size == 0:
		return np.zeros(4**k, dtype=np.uint32)

	pow4 = (4 ** np.arange(k-1, -1, -1, dtype=np.uint32))
	codes = (w.astype(np.uint32) * pow4).sum(axis=1)
	return np.bincount(codes, minlength=4**k).astype(np.uint32)


def normalize_rows(X: np.ndarray) -> np.ndarray:
	s = X.sum(axis=1, keepdims=True)
	s[s == 0] = 1
	return X / s


def prepare_input_batch(seqs, tokenizer, max_tokens=503):
	input_ids = []
	for seq in seqs:
		toks = tokenizer.tokenize(seq)
		toks = [tokenizer.cls_token] + toks[:max_tokens-3] + [tokenizer.sep_token, tokenizer.sep_token]
		ids = tokenizer.convert_tokens_to_ids(toks)
		input_ids.append(torch.tensor(ids, dtype=torch.long))
	pad_id = tokenizer.pad_token_id
	input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
	attn = (input_ids != pad_id).long()
	return input_ids, attn

@torch.no_grad()
def bert_cls_embeddings_batched(seqs, model, tokenizer, k, batch_size=256, max_tokens=503, device=None):
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.to(device).eval()
	hid = model.config.hidden_size
	out = np.empty((len(seqs), hid), dtype=np.float32)
	for i in range(0, len(seqs), batch_size):
		chunk = seqs[i:i+batch_size]
		tokenizer.k = k
		ids, mask = prepare_input_batch(chunk, tokenizer, max_tokens)
		ids = ids.to(device); mask = mask.to(device)
		outputs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
		cls = outputs.hidden_states[-1][:, 0, :].float().cpu().numpy()
		out[i:i+len(chunk)] = cls
	return out


def get_model(input_4mer_shape, input_5mer_shape, input_6mer_shape, num_class):
	input_a = tf.keras.Input(shape=input_4mer_shape, name="input_4")
	input_b = tf.keras.Input(shape=input_5mer_shape, name="input_5")
	input_c = tf.keras.Input(shape=input_6mer_shape, name="input_6")

	model1 = tf.keras.layers.Conv2D(100, (1, 3), activation='relu', input_shape=input_4mer_shape)(input_a)
	model1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(model1)
	model1 = tf.keras.layers.Dropout(0.5)(model1)
	model1 = tf.keras.layers.Flatten()(model1)

	model2 = tf.keras.layers.Conv2D(100, (1, 5), activation='relu', input_shape=input_5mer_shape)(input_b)
	model2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(model2)
	model2 = tf.keras.layers.Dropout(0.5)(model2)
	model2 = tf.keras.layers.Flatten()(model2)

	model3 = tf.keras.layers.Conv2D(100, (1, 7), activation='relu', input_shape=input_6mer_shape)(input_c)
	model3 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(model3)
	model3 = tf.keras.layers.Dropout(0.5)(model3)
	model3 = tf.keras.layers.Flatten()(model3)

	concat = tf.keras.layers.concatenate([model1, model2, model3], axis=1, name="concat_layer")
	output = tf.keras.layers.Dense(num_class, activation='softmax')(concat)

	model = tf.keras.Model(inputs=[input_a, input_b, input_c], outputs=[output])

	opt = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
	# loss function
	loss_fn = tf.keras.losses.CategoricalCrossentropy()
	# Compile model
	model.compile(loss=loss_fn, optimizer=opt, metrics=[f1_m])
	return model


def run_experiment(model, X_train, Y_train, X_dev, Y_dev, batch_size, num_epochs):
	lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_m', mode="max", factor=0.01, patience=10, verbose=1)
	early_stopping = EarlyStopping(monitor='val_f1_m', mode="max", patience=50, restore_best_weights=True)

	X_train = [X.astype("float32") for X in X_train]
	Y_train = tf.cast(Y_train, tf.float32)
	X_dev = [X.astype("float32") for X in X_dev]
	Y_dev = tf.cast(Y_dev, tf.float32)

	batch_size = min(batch_size, X_train[0].shape[0], X_dev[0].shape[0])
	train_steps = math.ceil(X_train[0].shape[0] / batch_size)
	val_steps = math.ceil(X_dev[0].shape[0] / batch_size)

	train_ds = (tf.data.Dataset.from_tensor_slices(((X_train[0], X_train[1], X_train[2]), Y_train))
				.shuffle(min(len(X_train[0]), 10000), reshuffle_each_iteration=True)
				.batch(batch_size, drop_remainder=False)
				.repeat()
				.prefetch(tf.data.AUTOTUNE))

	val_ds = (tf.data.Dataset.from_tensor_slices(((X_dev[0], X_dev[1], X_dev[2]), Y_dev))
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


def truncate_seq_indv(seq: str, L: int) -> str:
	if len(seq) <= L:
		return seq
	left = math.ceil(L / 2)
	right = L // 2
	return seq[:left] + seq[-right:]