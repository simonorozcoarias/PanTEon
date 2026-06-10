# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv1D
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import itertools
import re
import subprocess
import shutil
import math


##############################################
current_folder = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(current_folder, "..")

# 1. Data preprocessing parameters
## Whether to use corresponding features for classification, all of which have been proven helpful for classification.
use_kmers = 1   # Whether to use k-mer feature
use_terminal = 1    # Whether to use LTR and TIR features
use_TSD = 0     # Whether to use TSD feature
use_domain = 1  # Whether to use TE domain feature
use_ends = 1    # Whether to use 5-bp ends feature


use_minority = 0 # Whether to use minority samples to correct results
is_train = 0  # Whether it is in the model training stage
keep_raw = 0  # Whether to retain the raw input sequence, 1 yes, 0 no, only save species having TSDs
only_preprocess = 0 # Whether to only perform data preprocessing
is_predict = 1  # Enable prediction mode. Setting to 0 requires the input FASTA file to be in Repbase format (seq_name\tLabel\tspecies).
is_wicker = 1   # Use Wicker classification labels. Setting to 0 will output RepeatMasker classification labels.
is_plant = 0 # Is the input genome of a plant? 0 represents non-plant, while 1 represents plant.
is_debug = 0 # Is debug mode


# 2. Program and model parameters
internal_kmer_sizes = [1, 3]   # Size of k-mer used for converting internal sequences to k-mer frequency features
terminal_kmer_sizes = [1, 2, 3] # Size of k-mer used for converting terminal sequences to k-mer frequency features
## CNN model parameters
cnn_num_convs = 3 # Number of CNN convolutional layers
cnn_filters_array = [16, 16, 16] # Number of filters per convolutional layer in CNN
cnn_kernel_sizes_array = [7, 7, 7] # Kernel size for each convolutional layer in CNN; for 2D convolutional layers, set as [(3, 3), ...]
cnn_dropout = 0.5 # CNN dropout threshold
## Training parameters
batch_size = 32 # Batch size for training
epochs = 50 #  Number of epochs for training
use_checkpoint = 0  # Whether to use checkpoint training; set to 1 to resume training from the parameters of the previous failed training session, avoiding training from scratch


################################################### The following parameters do not need modification ######################################################################
version_num = '1.0.1'
work_dir = project_dir + '/work' # temp work directory

non_temp_files = ['classified\.info', 'classified_TE\.fa', '.*\.domain']

# minority sample labels
#minority_labels_class = {'Crypton': 0, '5S': 1, '7SL': 2, 'Merlin': 3, 'P': 4, 'R2': 5, 'Unknown': 6}
minority_labels_class = {'Crypton': 0, '5S': 1, 'Merlin': 2, 'P': 3, 'R2': 4, 'Unknown': 5}

## Superfamily labels based on Wicker classification original NeuralTE
all_wicker_class_original = {'Tc1-Mariner': 0, 'hAT': 1, 'Mutator': 2, 'Merlin': 3, 'Transib': 4, 'P': 5, 'PiggyBac': 6,
                    'PIF-Harbinger': 7, 'CACTA': 8, 'Crypton': 9, 'Helitron': 10, 'Maverick': 11, 'Copia': 12,
                    'Gypsy': 13, 'Bel-Pao': 14, 'Retrovirus': 15, 'DIRS': 16, 'Ngaro': 17, 'VIPER': 18,
                    'Penelope': 19, 'R2': 20, 'RTE': 21, 'Jockey': 22, 'L1': 23, 'I': 24, 'tRNA': 25, '7SL': 26, '5S': 27, 'Unknown': 28,
                    # New added superfamilies:
                    'LINE': 29, 'LTR': 30, 'SINE': 31, 'TIR': 32}

all_wicker_class = {'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
                   'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
                   'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
                   'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
                   'KOLOBOK': 28, 'ACADEM-1': 29, 'Unknown': 30}


## Augmentation for each Repbase data
expandClassNum = {'Merlin': 20, 'Transib': 10, 'P': 10, 'Crypton': 10, 'Penelope': 5, 'R2': 20, 'RTE': 8, 'Jockey': 10, 'I': 10}

class_num = len(all_wicker_class)
inverted_all_wicker_class = {value: key for key, value in all_wicker_class.items()}
# Maximum length of TSD (Target Site Duplication)
max_tsd_length = 15
# Obtain CNN input dimensions
X_feature_len = 0
# Dimensions of TE terminal and internal sequences
if use_kmers != 0:
    for kmer_size in internal_kmer_sizes:
        X_feature_len += pow(4, kmer_size)
    if use_terminal != 0:
        for i in range(2):
            for kmer_size in terminal_kmer_sizes:
                X_feature_len += pow(4, kmer_size)
if use_TSD != 0:
    X_feature_len += max_tsd_length * 4 + 1
# if use_minority != 0:
#     X_feature_len += len(minority_labels_class)
if use_domain != 0:
    X_feature_len += len(all_wicker_class_original)
if use_ends != 0:
    X_feature_len += 10 * 4


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


def get_model(work_dir, num_features, class_num):
    checkpoint_dir = work_dir + "/ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    os.system('cd ' + checkpoint_dir + ' && rm -rf ckpt*')
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return load_model(latest_checkpoint)
    print("Creating a new model")

    # CNN model
    # input layer
    input_layer = Input(shape=(num_features, 1))
    conv_input_layer = input_layer
    # Create multiple convolutional layers
    for i in range(cnn_num_convs):
        # Add convolutional layers
        conv = Conv1D(cnn_filters_array[i], cnn_kernel_sizes_array[i], activation='relu')(conv_input_layer)
        conv_input_layer = conv
    dropout1 = Dropout(0.5)(conv_input_layer)
    # Add flattening and fully connected layers
    flatten = Flatten()(dropout1)
    dense1 = Dense(128, activation='relu')(flatten)
    # Output layer
    output_layer = Dense(int(class_num), activation='softmax')(dense1)
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_m])

    return model


def run_experiment(model, X_train, Y_train, X_dev, Y_dev, batch_size, num_epochs):
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

    return history, None


def load_data(internal_kmer_sizes, terminal_kmer_sizes, data_path, work_dir, project_dir, threads):
    if not os.path.exists(data_path):
        print('Input file not exist: ' + data_path)
        exit(-1)
    os.makedirs(work_dir, exist_ok=True)
    shutil.copy2(data_path, work_dir)
    genome_info_path = work_dir + '/genome.info'
    data_path = work_dir + '/' + os.path.basename(data_path)
    domain_train_path = data_path + '.domain'

    minority_temp = work_dir + '/minority'
    if not os.path.exists(minority_temp):
        os.makedirs(minority_temp)
    minority_train_path = minority_temp + '/train.minority.ref'
    minority_out = minority_temp + '/train.minority.out'

    data_path = preprocess_data(data_path, domain_train_path, minority_train_path, minority_out, work_dir,
                                     project_dir, project_dir+"/tools/", threads, 0)

    X, Y, seq_names, labels = load_repbase_with_TSD(data_path, domain_train_path, minority_train_path, minority_out,
                                            all_wicker_class_original, project_dir + '/data/TEClasses.tsv')

    X, Y = generate_feature_mats(X, Y, seq_names, minority_labels_class, all_wicker_class,
                                 internal_kmer_sizes, terminal_kmer_sizes, threads, all_wicker_class_original)

    # Reshape data into the format accepted by the model
    X = X.reshape(X.shape[0], X_feature_len, 1).astype('float32', copy=False)
    return X, Y, seq_names, data_path, labels


def preprocess_data(data, domain_train_path, minority_train_path, minority_out, work_dir, project_dir,
                    tool_dir, threads, is_train):
    # Delete previous run's retained results
    SegLTR2intactLTRMap = work_dir + '/segLTR2intactLTR.map'
    os.system('rm -f ' + SegLTR2intactLTRMap)

    generate_domain_info(data, project_dir + '/data/RepeatPeps.lib', work_dir, threads)
    generate_minority_info(data, minority_train_path, minority_out, threads, is_train)
    data = generate_terminal_info(data, work_dir, tool_dir, threads)
    return data


##################### utils.data_util.py
##word_seq generates eg. ['AA', 'AT', 'TC', 'CG', 'GT']
def word_seq(seq, k, stride=1):
    i = 0
    words_list = []
    while i <= len(seq) - k:
        words_list.append(seq[i: i + k])
        i += stride
    return (words_list)

def generate_kmer_dic(repeat_num):
    kmer_dic = {}
    bases = ['A','G','C','T']
    kmer_list = list(itertools.product(bases, repeat=int(repeat_num)))
    for eachitem in kmer_list:
        #print(eachitem)
        each_kmer = ''.join(eachitem)
        kmer_dic[each_kmer] = 0

    return (kmer_dic)

def generate_mat(words_list,kmer_dic):
    for eachword in words_list:
        kmer_dic[eachword] += 1
    num_list = []
    for eachkmer in kmer_dic:
        num_list.append(kmer_dic[eachkmer])
    return (num_list)


BASE_TO_INT = np.frombuffer(bytearray(b"AGCT"), dtype=np.uint8)
MAP = {ord('A'):0, ord('G'):1, ord('C'):2, ord('T'):3}

def encode_seq_np(seq):
    a = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
    out = np.full(a.shape, 255, dtype=np.uint8)  # 255 = N/otros
    for ch, val in MAP.items():
        out[a == ch] = val
    return out


def kmer_counts_fast(enc, k):
    if enc.size < k:
        return np.zeros(4**k, dtype=np.int32)

    power = 4**(k-1)
    idx = enc[:k].copy()
    if (idx==255).any():
        cur = -1
    else:
        cur = 0
        for v in idx: cur = cur*4 + v
    counts = np.zeros(4**k, dtype=np.int32)
    bad = 0
    for i in range(k, enc.size+1):
        if cur >= 0:
            counts[cur] += 1
        if i == enc.size: break
        left = enc[i-k]
        right = enc[i]
        if right == 255:
            bad = k
        elif bad > 0:
            bad -= 1
        if bad > 0 or left == 255:
            cur = -1
        else:
            if cur < 0:
                window = enc[i-k+1:i+1]
                if (window==255).any():
                    cur = -1
                else:
                    cur = 0
                    for v in window: cur = cur*4 + v
            else:
                cur = (cur - left*power)*4 + right
    return counts


def get_batch_kmer_freq_v1(grouped_x, internal_kmer_sizes, terminal_kmer_sizes, minority_labels_class, all_wicker_class_original):
    BASE_ORDER = 'AGCT'
    BASE_MAP = {ord(b): i for i, b in enumerate(BASE_ORDER)}
    BAD = 255

    def encode_seq_np(seq: str) -> np.ndarray:
        a = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        out = np.full(a.shape, BAD, dtype=np.uint8)
        for ch, val in BASE_MAP.items():
            out[a == ch] = val
        return out

    def kmer_counts_fast(enc: np.ndarray, k: int) -> np.ndarray:
        L = enc.size
        if L < k:
            return np.zeros(4 ** k, dtype=np.int32)
        power = 4 ** (k - 1)
        counts = np.zeros(4 ** k, dtype=np.int32)
        w = enc[:k]
        if (w == BAD).any():
            cur = -1
            bad = k
        else:
            cur = 0
            for v in w: cur = cur * 4 + int(v)
            bad = 0
        for i in range(k, L + 1):
            if cur >= 0:
                counts[cur] += 1
            if i == L:
                break
            left = int(enc[i - k]);
            right = int(enc[i])
            if right == BAD:
                bad = k
            elif bad > 0:
                bad -= 1
            if bad > 0 or left == BAD:
                cur = -1
                if bad == 0:
                    w = enc[i - k + 1:i + 1]
                    if (w == BAD).any():
                        cur = -1
                        bad = k
                    else:
                        cur = 0
                        for v in w: cur = cur * 4 + int(v)
            else:
                cur = (cur - left * power) * 4 + right
        return counts

    global use_kmers, use_terminal, use_TSD, use_ends, use_domain, max_tsd_length
    internal_len = sum(4 ** k for k in internal_kmer_sizes) if use_kmers else 0
    term_len = (2 * sum(4 ** k for k in terminal_kmer_sizes)) if (use_kmers and use_terminal) else 0
    tsd_len = (1 + max_tsd_length * 4) if use_TSD else 0
    ends_len = (10 * 4) if use_ends else 0
    domain_len = len(all_wicker_class_original) if use_domain else 0
    total_len = internal_len + term_len + tsd_len + ends_len + domain_len

    group_dict = {}
    for x in grouped_x:
        seq_name = x[0]
        seq = x[1]
        TSD_seq = x[2]
        TSD_len = x[3]
        LTR_pos = x[4]
        TIR_pos = x[5]
        domain_label_set = x[6]

        internal_seq = ''
        LTR_seq = ''
        TIR_seq = ''
        LTR_pos_str = str(LTR_pos.split(':')[1]).strip()
        TIR_pos_str = str(TIR_pos.split(':')[1]).strip()
        if LTR_pos_str == '' and TIR_pos_str == '':
            internal_seq = seq
        if TIR_pos_str != '':
            T = TIR_pos_str.split(',')
            left_TIR_start = int(T[0].split('-')[0])
            left_TIR_end = int(T[0].split('-')[1])
            right_TIR_start = int(T[1].split('-')[0])
            right_TIR_end = int(T[1].split('-')[1])
            TIR_seq = seq[left_TIR_start - 1: left_TIR_end] + seq[right_TIR_start - 1: right_TIR_end]
            internal_seq = seq[left_TIR_end: right_TIR_start - 1]
        if LTR_pos_str != '':
            L = LTR_pos_str.split(',')
            left_LTR_start = int(L[0].split('-')[0])
            left_LTR_end = int(L[0].split('-')[1])
            right_LTR_start = int(L[1].split('-')[0])
            right_LTR_end = int(L[1].split('-')[1])
            LTR_seq = seq[left_LTR_start - 1: left_LTR_end]
            internal_seq = seq[left_LTR_end: right_LTR_start - 1]

        connected = np.zeros(total_len, dtype=np.int32)
        offset = 0

        if use_kmers:
            target_seq = internal_seq if use_terminal else seq
            enc = encode_seq_np(target_seq) if target_seq else np.array([], dtype=np.uint8)
            for k in internal_kmer_sizes:
                cnt = kmer_counts_fast(enc, k) if enc.size else np.zeros(4 ** k, dtype=np.int32)
                n = cnt.size
                connected[offset:offset + n] = cnt;
                offset += n

            if use_terminal:
                enc_LTR = encode_seq_np(LTR_seq) if LTR_seq else np.array([], dtype=np.uint8)
                enc_TIR = encode_seq_np(TIR_seq) if TIR_seq else np.array([], dtype=np.uint8)
                for k in terminal_kmer_sizes:
                    cnt = kmer_counts_fast(enc_LTR, k) if enc_LTR.size else np.zeros(4 ** k, dtype=np.int32)
                    n = cnt.size
                    connected[offset:offset + n] = cnt;
                    offset += n
                    cnt = kmer_counts_fast(enc_TIR, k) if enc_TIR.size else np.zeros(4 ** k, dtype=np.int32)
                    n = cnt.size
                    connected[offset:offset + n] = cnt;
                    offset += n

        if use_TSD:
            max_length = max_tsd_length
            if (TSD_seq == 'Unknown') or ('N' in TSD_seq):
                connected[offset] = max_length + 1;
                offset += 1
                encoded_TSD = np.ones((max_length, 4), dtype=np.int8)
            else:
                connected[offset] = int(TSD_len);
                offset += 1
                encoded_TSD = np.zeros((max_length, 4), dtype=np.int8)
                L = min(len(TSD_seq), max_length)
                eye = np.eye(4, dtype=np.int8)
                for i in range(L):
                    b = TSD_seq[i]
                    if b == 'A':
                        encoded_TSD[i] = eye[0]
                    elif b == 'T':
                        encoded_TSD[i] = eye[1]
                    elif b == 'C':
                        encoded_TSD[i] = eye[2]
                    elif b == 'G':
                        encoded_TSD[i] = eye[3]
            n = max_length * 4
            connected[offset:offset + n] = encoded_TSD.reshape(-1);
            offset += n

        if use_ends:
            end_seq = (seq[:5] + seq[-5:]) if len(seq) >= 10 else (seq + 'N' * (10 - len(seq)))
            eye = np.eye(4, dtype=np.int8)
            encoded_end_seq = np.zeros((10, 4), dtype=np.int8)
            for i, base in enumerate(end_seq[:10]):
                if base == 'A':
                    encoded_end_seq[i] = eye[0]
                elif base == 'T':
                    encoded_end_seq[i] = eye[1]
                elif base == 'C':
                    encoded_end_seq[i] = eye[2]
                elif base == 'G':
                    encoded_end_seq[i] = eye[3]
            n = 10 * 4
            connected[offset:offset + n] = encoded_end_seq.reshape(-1);
            offset += n

        if use_domain:
            encoder = np.zeros(domain_len, dtype=np.int32)
            for domain_label in domain_label_set:
                domain_label_num = all_wicker_class_original[domain_label]
                encoder[domain_label_num] = 1
            n = domain_len
            connected[offset:offset + n] = encoder
            offset += n

        group_dict[seq_name] = connected

    return group_dict


def split_list_into_groups(lst, group_size):
    return [lst[i:i+group_size] for i in range(0, len(lst), group_size)]


def generate_feature_mats(X, Y, seq_names, minority_labels_class, all_wicker_class, internal_kmer_sizes, terminal_kmer_sizes, threads, all_wicker_class_original):
    seq_mats = {}
    jobs = []
    grouped_X = split_list_into_groups(X, 5000)

    ex = ProcessPoolExecutor(threads)

    for grouped_x in grouped_X:
        job = ex.submit(get_batch_kmer_freq_v1, grouped_x, internal_kmer_sizes, terminal_kmer_sizes, minority_labels_class, all_wicker_class_original)
        jobs.append(job)
    ex.shutdown(wait=True)

    for job in as_completed(jobs):
        cur_group_dict = job.result()
        seq_mats.update(cur_group_dict)

    final_X = []
    final_Y = []
    for item in seq_names:
        seq_name = item[0]
        x = seq_mats[seq_name]
        final_X.append(x)
        label = Y[seq_name]
        label_num = all_wicker_class[label]
        final_Y.append(label_num)
    return np.array(final_X), np.array(final_Y)


def replace_non_atcg(sequence):
    return re.sub("[^ATCG]", "", sequence)


def getRMToWicker(RM_Wicker_struct):
    rmToWicker = {}
    wicker_superfamily_set = set()
    with open(RM_Wicker_struct, 'r') as f_r:
        for i, line in enumerate(f_r):
            parts = line.split('\t')
            rm_type = parts[5]
            rm_subtype = parts[6]
            repbase_type = parts[7]
            wicker_type = parts[8]
            wicker_type_parts = wicker_type.split('/')
            wicker_superfamily_parts = wicker_type_parts[-1].strip().split(' ')
            if len(wicker_superfamily_parts) == 1:
                wicker_superfamily = wicker_superfamily_parts[0]
            elif len(wicker_superfamily_parts) > 1:
                wicker_superfamily = wicker_superfamily_parts[1].replace('(', '').replace(')', '')
            rm_full_type = rm_type + '/' + rm_subtype
            if wicker_superfamily == 'ERV':
                wicker_superfamily = 'Retrovirus'
            rmToWicker[rm_full_type] = wicker_superfamily
            wicker_superfamily_set.add(wicker_superfamily)

    rmToWicker['LINE/R2'] = 'R2'
    rmToWicker['LINE/RTE'] = 'RTE'
    rmToWicker['LTR/ERVL'] = 'Retrovirus'
    rmToWicker['LTR/Ngaro'] = 'DIRS'
    return rmToWicker


def load_repbase_with_TSD(path, domain_path, minority_train_path, minority_out, all_wicker_class_original, RM_Wicker_struct):
    rmToWicker = getRMToWicker(RM_Wicker_struct)
    domain_name_labels = {}
    if use_domain == 1 and os.path.exists(domain_path):

        with open(domain_path, 'r') as f_r:
            for i, line in enumerate(f_r):
                if i < 2:
                    continue
                parts = line.split('\t')
                TE_name = parts[0]
                label = parts[1].split('#')[1]
                if not rmToWicker.__contains__(label):
                    label = 'Unknown'
                else:
                    wicker_superfamily = rmToWicker[label]
                    label = wicker_superfamily
                    if not all_wicker_class_original.__contains__(label):
                        label = 'Unknown'
                if not domain_name_labels.__contains__(TE_name):
                    domain_name_labels[TE_name] = set()
                label_set = domain_name_labels[TE_name]
                label_set.add(label)

    names, contigs = read_fasta_v1(path)
    X = []
    Y = {}
    seq_names = []
    labels = []
    for name in names:
        feature_info = {}
        parts = name.split("\t")
        seq_name = parts[0].split(" ")[0]
        if "#" in parts[0].split(" ")[0]:
            label = parts[0].split(" ")[0].split("#")[1]
        else:
            label = "Unknown"
        labels.append(parts[0].split(" ")[0])
        for p_name in parts:
            if 'TSD:' in p_name:
                TSD_seq = p_name.split(':')[1]
                feature_info['TSD_seq'] = TSD_seq
            elif 'TSD_len:' in p_name:
                tsd_len_str = p_name.split(':')[1]
                if tsd_len_str == '':
                    TSD_len = 0
                else:
                    TSD_len = int(tsd_len_str)
                feature_info['TSD_len'] = TSD_len
            elif 'LTR:' in p_name:
                LTR_info = p_name
                feature_info['LTR_info'] = LTR_info
            elif 'TIR:' in p_name:
                TIR_info = p_name
                feature_info['TIR_info'] = TIR_info
        if use_TSD:
            TSD_seq = feature_info['TSD_seq']
            TSD_len = feature_info['TSD_len']
        else:
            TSD_seq = ''
            TSD_len = 0

        if use_terminal:
            LTR_info = feature_info['LTR_info']
            TIR_info = feature_info['TIR_info']
        else:
            LTR_info = 'LTR:'
            TIR_info = 'TIR:'

        if seq_name.endswith('-RC'):
            raw_seq_name = seq_name[:-3]
        else:
            raw_seq_name = seq_name
        if domain_name_labels.__contains__(raw_seq_name):
            domain_label_set = domain_name_labels[raw_seq_name]
        else:
            domain_label_set = {'Unknown'}

        seq = contigs[name]
        seq = replace_non_atcg(seq)  # undetermined nucleotides in splice
        x_feature = (seq_name, seq, TSD_seq, TSD_len, LTR_info, TIR_info, domain_label_set)
        X.append(x_feature)
        Y[seq_name] = label
        seq_names.append((seq_name, label))
    return X, Y, seq_names, labels


def split_fasta(cur_path, output_dir, num_chunks):
    split_files = []

    if os.path.exists(output_dir):
        os.system('rm -rf ' + output_dir)
    os.makedirs(output_dir)

    names, contigs = read_fasta_v1(cur_path)
    num_names = len(names)
    chunk_size = num_names // num_chunks

    for i in range(num_chunks):
        chunk_start = i * chunk_size
        chunk_end = chunk_start + chunk_size if i < num_chunks - 1 else num_names
        chunk = names[chunk_start:chunk_end]
        output_path = output_dir + '/out_' + str(i) + '.fa'
        with open(output_path, 'w') as out_file:
            for name in chunk:
                seq = contigs[name]
                out_file.write('>'+name+'\n'+seq+'\n')
        split_files.append(output_path)
    return split_files


def run_command(command):
    subprocess.run(command, check=True, shell=True)


def identify_terminals(split_file, output_dir, tool_dir):
    base_file = os.path.basename(split_file)
    try:
        ltrsearch_command = 'cd ' + output_dir + ' && ' + tool_dir + '/ltrsearch -l 50 ' + base_file + ' > /dev/null 2>&1'
        itrsearch_command = 'cd ' + output_dir + ' && ' + tool_dir + '/itrsearch -i 0.7 -l 7 ' + base_file + ' > /dev/null 2>&1'
        run_command(ltrsearch_command)
        run_command(itrsearch_command)
        ltr_file = split_file + '.ltr'
        tir_file = split_file + '.itr'

        # Read ltr and itr files to get the start and end positions of ltr and itr.
        ltr_names, ltr_contigs = read_fasta_v1(ltr_file)
        tir_names, tir_contigs = read_fasta_v1(tir_file)
        LTR_info = {}
        for ltr_name in ltr_names:
            parts = ltr_name.split(' ')
            orig_name = parts[0] + " " + parts[1]
            terminal_info = " ".join(parts[1:])
            LTR_info_parts = terminal_info.split('LTR')[1].split(' ')[0].replace('(', '').replace(')', '').split('..')
            LTR_left_pos_parts = LTR_info_parts[0].split(',')
            LTR_right_pos_parts = LTR_info_parts[1].split(',')
            lLTR_start = int(LTR_left_pos_parts[0])
            lLTR_end = int(LTR_left_pos_parts[1])
            rLTR_start = int(LTR_right_pos_parts[0])
            rLTR_end = int(LTR_right_pos_parts[1])
            LTR_info[orig_name] = (lLTR_start, lLTR_end, rLTR_start, rLTR_end)
        TIR_info = {}
        for tir_name in tir_names:
            parts = tir_name.split(' ')
            orig_name = parts[0] + " " + parts[1]
            terminal_info = " ".join(parts[1:])
            TIR_info_parts = terminal_info.split('ITR')[1].split(' ')[0].replace('(', '').replace(')', '').split('..')
            TIR_left_pos_parts = TIR_info_parts[0].split(',')
            TIR_right_pos_parts = TIR_info_parts[1].split(',')
            lTIR_start = int(TIR_left_pos_parts[0])
            lTIR_end = int(TIR_left_pos_parts[1])
            rTIR_start = int(TIR_right_pos_parts[1])
            rTIR_end = int(TIR_right_pos_parts[0])
            TIR_info[orig_name] = (lTIR_start, lTIR_end, rTIR_start, rTIR_end)

        # Update the header of the split_file, adding two columns LTR:1-206,4552-4757 TIR:1-33,3869-3836.
        update_split_file = split_file + '.updated'
        update_contigs = {}
        names, contigs = read_fasta_v1(split_file)
        for name in names:
            orig_name = name
            LTR_str = 'LTR:'
            if LTR_info.__contains__(orig_name):
                lLTR_start, lLTR_end, rLTR_start, rLTR_end = LTR_info[orig_name]
                LTR_str += str(lLTR_start) + '-' + str(lLTR_end) + ',' + str(rLTR_start) + '-' + str(rLTR_end)
            TIR_str = 'TIR:'
            if TIR_info.__contains__(orig_name):
                lTIR_start, lTIR_end, rTIR_start, rTIR_end = TIR_info[orig_name]
                TIR_str += str(lTIR_start) + '-' + str(lTIR_end) + ',' + str(rTIR_start) + '-' + str(rTIR_end)
            update_name = name + '\t' + LTR_str + '\t' + TIR_str
            update_contigs[update_name] = contigs[name]
        store_fasta(update_contigs, update_split_file)

        return update_split_file
    except Exception as e:
        print(f"Error processing file {split_file} .....")
        return e


def generate_terminal_info(data_path, work_dir, tool_dir, threads):
    output_dir = work_dir + '/temp'
    split_files = split_fasta(data_path, output_dir, threads)

    # Parallelize the identification of LTR and TIR.
    cur_update_path = data_path + '.update'
    os.system('rm -f ' + cur_update_path)
    with ProcessPoolExecutor(threads) as executor:
        futures = []
        for split_file in split_files:
            future = executor.submit(identify_terminals, split_file, output_dir, tool_dir)
            futures.append(future)
        executor.shutdown(wait=True)

        is_exit = False
        for future in as_completed(futures):
            update_split_file = future.result()
            if isinstance(update_split_file, str):
                os.system('cat ' + update_split_file + ' >> ' + cur_update_path)
            else:
                print(f"An error occurred: {update_split_file}")
                is_exit = True
                break
        if is_exit:
            print('Error occur, exit...')
            exit(1)
        else:
            shutil.move(cur_update_path, data_path)

    return data_path


def generate_domain_info(input_path, domain_path, work_dir, threads):
    output_table = input_path + '.domain'
    temp_dir = work_dir + '/domain'
    get_domain_info(input_path, domain_path, output_table, threads, temp_dir)


def generate_minority_info(train_path, minority_train_path, minority_out, threads, is_train):
    if is_train:
        minority_contigs = {}
        train_contigNames, train_contigs = read_fasta_v1(train_path)
        # 1. extract minority dataset
        for name in train_contigNames:
            label = name.split('\t')[1]
            if minority_labels_class.__contains__(label):
                minority_contigs[name] = train_contigs[name]
        store_fasta(minority_contigs, minority_train_path)


def store2file(data_partition, cur_consensus_path):
    if len(data_partition) > 0:
        with open(cur_consensus_path, 'w') as f_save:
            for item in data_partition:
                f_save.write('>'+item[0]+'\n'+item[1]+'\n')
        f_save.close()


def PET(seq_item, partitions):
    # sort contigs by length
    original = seq_item
    original = sorted(original, key=lambda x: len(x[1]), reverse=True)
    return divided_array(original, partitions)


def divided_array(original_array, partitions):
    final_partitions = [[] for _ in range(partitions)]
    node_index = 0

    read_from_start = True
    read_from_end = False
    i = 0
    j = len(original_array) - 1
    while i <= j:
        # read from file start
        if read_from_start:
            final_partitions[node_index % partitions].append(original_array[i])
            i += 1
        if read_from_end:
            final_partitions[node_index % partitions].append(original_array[j])
            j -= 1
        node_index += 1
        if node_index % partitions == 0:
            # reverse
            read_from_end = bool(1 - read_from_end)
            read_from_start = bool(1 - read_from_start)
    return final_partitions


def get_domain_info(cons, lib, output_table, threads, temp_dir):
    if os.path.exists(temp_dir):
        os.system('rm -rf ' + temp_dir)
    os.makedirs(temp_dir)

    consensus_contignames, consensus_contigs = read_fasta_v1(cons)
    # Copy the lib to the output directory. If the current process involves
    # evaluation, then it's necessary to filter out domains from lib
    # that contain test species
    temp_lib = temp_dir + '/RepeatPeps.lib'
    shutil.copy2(lib, temp_lib)
    if is_predict == 0:
        test_species_set = set()
        for name in consensus_contignames:
            parts = name.split('\t')
            species = parts[2]
            test_species_set.add(species)
        # filter out test species from the protein library of RepeatMasker.
        lib_contigNames, lib_contigs = read_fasta_v1(lib)
        rm_contigs = {}
        for name in lib_contigNames:
            pattern = r'\[(.*?)\]'
            match = re.search(pattern, name)
            if match:
                species = match.group(1)
            else:
                species = 'Unknown'
            # filter content in '()'
            pattern = r'\([^)]*\)'
            species = re.sub(pattern, '', species)
            species = re.sub(r'\s+', ' ', species).strip()
            if species not in test_species_set:
                rm_contigs[name] = lib_contigs[name]
        store_fasta(rm_contigs, temp_lib)

    lib = temp_lib
    blast_db_command = 'makeblastdb -dbtype prot -in ' + lib
    os.system(blast_db_command + ' > /dev/null 2>&1')
    # 1. Divide the cons, and for each block, use blastx -num_threads 1 -evalue 1e-20 to compare cons with domain.
    partitions_num = int(threads)
    data_partitions = PET(consensus_contigs.items(), partitions_num)
    merge_distance = 100
    file_list = []
    ex = ProcessPoolExecutor(threads)
    jobs = []
    for partition_index, data_partition in enumerate(data_partitions):
        if len(data_partition) <= 0:
            continue
        cur_consensus_path = temp_dir + '/'+str(partition_index)+'.fa'
        store2file(data_partition, cur_consensus_path)
        cur_output = temp_dir + '/'+str(partition_index)+'.out'
        cur_table = temp_dir + '/' + str(partition_index) + '.tbl'
        cur_file = (cur_consensus_path, lib, cur_output, cur_table)

        job = ex.submit(multiple_alignment_blastx_v1, cur_file, merge_distance)
        jobs.append(job)
    ex.shutdown(wait=True)

    # 2. Generate a table of the best matches between query and domain.
    os.system("echo 'TE_name\tdomain_name\tTE_start\tTE_end\tdomain_start\tdomain_end\n' > " + output_table)
    is_exit = False
    for job in as_completed(jobs):
        cur_table = job.result()
        if isinstance(cur_table, str):
            os.system('cat ' + cur_table + ' >> ' + output_table)
        else:
            print(f"An error occurred: {cur_table}")
            is_exit = True
            break
    if is_exit:
        print('Error occur, exit...')
        exit(1)


def multiple_alignment_blastx_v1(repeats_path, merge_distance):
    try:
        split_repeats_path = repeats_path[0]
        protein_db_path = repeats_path[1]
        blastx2Results_path = repeats_path[2]
        cur_table = repeats_path[3]
        align_command = 'blastx -db ' + protein_db_path + ' -num_threads ' \
                        + str(1) + ' -evalue 1e-20 -query ' + split_repeats_path + ' -outfmt 6 > ' + blastx2Results_path
        run_command(align_command)

        fixed_extend_base_threshold = merge_distance

        query_records = {}
        with open(blastx2Results_path, 'r') as f_r:
            for idx, line in enumerate(f_r):
                # print('current line idx: %d' % (idx))
                parts = line.split('\t')
                query_name = parts[0]
                subject_name = parts[1]
                identity = float(parts[2])
                alignment_len = int(parts[3])
                q_start = int(parts[6])
                q_end = int(parts[7])
                s_start = int(parts[8])
                s_end = int(parts[9])
                if not query_records.__contains__(query_name):
                    query_records[query_name] = {}
                subject_dict = query_records[query_name]
                if not subject_dict.__contains__(subject_name):
                    subject_dict[subject_name] = []
                cur_records = subject_dict[subject_name]
                # q_start, q_end, s_start, s_end
                if q_start <= q_end:
                    cur_records.append((q_start, q_end, s_start, s_end))
                else:
                    cur_records.append((q_end, q_start, s_start, s_end))

        # remove redundant records
        keep_longest_query = {}
        for query_name in query_records.keys():
            keep_longest_query[query_name] = []

            subject_dict = query_records[query_name]

            # forward and reverse respectively, cluster
            # pos --> [q_start, q_end, s_start, s_end]
            # reverse --> [q_start, q_end, s_end, s_start]
            pos_array = []
            reverse_array = []
            for subject_name in subject_dict.keys():
                cur_records = subject_dict[subject_name]
                cur_pos = []
                cur_reverse = []
                for frag in cur_records:
                    q_start = frag[0]
                    q_end = frag[1]
                    s_start = frag[2]
                    s_end = frag[3]
                    if s_start <= s_end:
                        cur_pos.append([q_start, q_end, s_start, s_end, subject_name])
                    else:
                        cur_reverse.append([q_start, q_end, s_end, s_start, subject_name])

                # sort by q_start
                cur_pos.sort(key=lambda x: (x[0], x[1]))
                cur_reverse.sort(key=lambda x: (x[0], x[1]))

                # cluster
                pos_array.append(cur_pos)
                reverse_array.append(cur_reverse)

            # print('len(pos_array): %d' % (len(pos_array)))

            merge_domains = []
            # forward strand
            for pos in pos_array:
                clusters = {}
                cluster_index = 0
                if len(pos) > 0:
                    cur_cluster = []
                    cur_cluster.append(pos[0])
                    clusters[cluster_index] = cur_cluster
                    for i in range(1, len(pos)):
                        frag = pos[i]
                        cur_cluster = clusters[cluster_index]
                        is_closed = False
                        for exist_frag in reversed(cur_cluster):
                            if (frag[0] - exist_frag[1] < fixed_extend_base_threshold):
                                is_closed = True
                                break
                        if is_closed:
                            cur_cluster.append(frag)
                        else:
                            cluster_index += 1
                            if not clusters.__contains__(cluster_index):
                                clusters[cluster_index] = []
                            cur_cluster = clusters[cluster_index]
                            cur_cluster.append(frag)

                for cluster_index in clusters.keys():
                    cur_cluster = clusters[cluster_index]
                    cur_cluster.sort(key=lambda x: (x[2], x[3]))

                    cluster_longest_query_start = -1
                    cluster_longest_query_end = -1
                    cluster_longest_subject_start = -1
                    cluster_longest_subject_end = -1
                    subject_name = ''
                    if len(cur_cluster) > 0:
                        cluster_longest_query_start = cur_cluster[0][0]
                        cluster_longest_subject_start = cur_cluster[0][2]
                        subject_name = cur_cluster[0][4]
                        for frag in cur_cluster:
                            cluster_longest_query_end = max(cluster_longest_query_end, frag[1])
                            cluster_longest_subject_end = max(cluster_longest_subject_end, frag[3])

                    if cluster_longest_query_start >= 0:
                        domain_len = cluster_longest_query_end - cluster_longest_query_start + 1
                        subject_len = cluster_longest_subject_end - cluster_longest_subject_start + 1
                        merge_domains.append([cluster_longest_query_start, cluster_longest_query_end,
                                              domain_len, cluster_longest_subject_start, cluster_longest_subject_end,
                                              subject_len, subject_name])

            # reverse strand
            for reverse_pos in reverse_array:
                clusters = {}
                cluster_index = 0
                if len(reverse_pos) > 0:
                    cur_cluster = []
                    cur_cluster.append(reverse_pos[0])
                    clusters[cluster_index] = cur_cluster
                    for i in range(1, len(reverse_pos)):
                        frag = reverse_pos[i]
                        cur_cluster = clusters[cluster_index]
                        is_closed = False
                        for exist_frag in reversed(cur_cluster):
                            if (exist_frag[1] - frag[0] < fixed_extend_base_threshold):
                                is_closed = True
                                break
                        if is_closed:
                            cur_cluster.append(frag)
                        else:
                            cluster_index += 1
                            if not clusters.__contains__(cluster_index):
                                clusters[cluster_index] = []
                            cur_cluster = clusters[cluster_index]
                            cur_cluster.append(frag)

                for cluster_index in clusters.keys():
                    cur_cluster = clusters[cluster_index]
                    cur_cluster.sort(key=lambda x: (x[2], x[3]))

                    cluster_longest_query_start = -1
                    cluster_longest_query_end = -1
                    cluster_longest_subject_start = -1
                    cluster_longest_subject_end = -1
                    subject_name = ''
                    if len(cur_cluster) > 0:
                        cluster_longest_query_start = cur_cluster[0][0]
                        cluster_longest_subject_start = cur_cluster[0][2]
                        subject_name = cur_cluster[0][4]
                        for frag in cur_cluster:
                            cluster_longest_query_end = max(cluster_longest_query_end, frag[1])
                            cluster_longest_subject_end = max(cluster_longest_subject_end, frag[3])

                    if cluster_longest_query_start >= 0:
                        domain_len = cluster_longest_query_end - cluster_longest_query_start + 1
                        subject_len = cluster_longest_subject_end - cluster_longest_subject_start + 1
                        merge_domains.append([cluster_longest_query_start, cluster_longest_query_end,
                                              domain_len, cluster_longest_subject_start, cluster_longest_subject_end,
                                              subject_len, subject_name])

            # remove redundant domains
            # keep_longest_query --> [[left, right, q_len, s_left, s_right, s_len, subject], ...]
            keep_domains = []
            merge_domains.sort(key=lambda x: (x[0], -x[1]))
            for i in range(len(merge_domains)):
                domain_i = merge_domains[i]
                is_new_domain = True
                for j in range(i):
                    domain_j = merge_domains[j]
                    left = max(domain_i[0], domain_j[0])
                    right = min(domain_i[1], domain_j[1])
                    if right >= left:
                        # if more than 50% overlapped with the previous longer one, drop it
                        overlap = right - left + 1
                        len_i = domain_i[1] - domain_i[0] + 1
                        if (overlap / len_i) > 0.5:
                            is_new_domain = False
                            break
                if is_new_domain:
                    keep_domains.append(domain_i)

            keep_longest_query[query_name] = keep_domains

        # Save table
        with open(cur_table, 'w') as f_save:
            for query_name in keep_longest_query.keys():
                domain_array = keep_longest_query[query_name]
                merge_domains = []
                for domain_info in domain_array:
                    is_new_domain = True
                    for k in range(len(merge_domains)):
                        exist_domain = merge_domains[k]
                        left = max(exist_domain[0], domain_info[0])
                        right = min(exist_domain[1], domain_info[1])
                        if right >= left:
                            overlap = right - left + 1
                            len_i = domain_info[1] - domain_info[0] + 1
                            if (overlap / len_i) > 0.5:
                                is_new_domain = False
                                break
                    if is_new_domain:
                        merge_domains.append(domain_info)

                for domain_info in merge_domains:
                    domain_name = str(domain_info[6]).replace(',', '')
                    f_save.write(query_name + '\t' + domain_name + '\t' +
                                 str(domain_info[0]) + '\t' + str(domain_info[1]) + '\t' +
                                 str(domain_info[3]) + '\t' + str(domain_info[4]) + '\n')

        return cur_table
    except Exception as e:
        return e


def read_fasta(fasta_path):
    contignames = []
    contigs = {}
    if os.path.exists(fasta_path):
        with open(fasta_path, 'r') as rf:
            contigname = ''
            contigseq = ''
            for line in rf:
                if line.startswith('>'):
                    if contigname != '' and contigseq != '':
                        contigs[contigname] = contigseq
                        contignames.append(contigname)
                    contigname = line.strip()[1:].split(" ")[0].split('\t')[0]
                    contigseq = ''
                else:
                    contigseq += line.strip().upper()
            if contigname != '' and contigseq != '':
                contigs[contigname] = contigseq
                contignames.append(contigname)
        rf.close()
    return contignames, contigs


def read_fasta_v1(fasta_path):
    contignames = []
    contigs = {}
    if os.path.exists(fasta_path):
        with open(fasta_path, 'r') as rf:
            contigname = ''
            contigseq = ''
            for line in rf:
                if line.startswith('>'):
                    if contigname != '' and contigseq != '':
                        contigs[contigname] = contigseq
                        contignames.append(contigname)
                    contigname = line.strip()[1:]
                    contigseq = ''
                else:
                    contigseq += line.strip().upper()
            if contigname != '' and contigseq != '':
                contigs[contigname] = contigseq
                contignames.append(contigname)
        rf.close()
    return contignames, contigs


def store_fasta(contigs, file_path):
    with open(file_path, 'w') as f_save:
        for name in contigs.keys():
            seq = contigs[name]
            f_save.write('>'+name+'\n'+seq+'\n')
    f_save.close()
