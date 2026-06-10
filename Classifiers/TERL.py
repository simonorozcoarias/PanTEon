from Bio import SeqIO
import numpy as np
import time
import tensorflow as tf
import datetime
import os
import seaborn as sn
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("[INFO] Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import metrics as sk_m
from typing import List

class Metric:
    def __init__(self,
                    labels: List[int],
                    predictions: List[int],
                    classes: List[str]=[],
                    f_beta: float=1.0,
                    cm: List[List[int]]=[],
                    output_dir: str='./Outputs',
                    filename_prefix: str=''):
        self.labels = labels
        self.predictions = predictions
        self.output_dir=output_dir
        self.filename_prefix=filename_prefix
        self.predictions = predictions
        self.f_beta = f_beta
        if cm == []:
            self.cm = sk_m.confusion_matrix(self.labels, self.predictions)
        else:
            self.cm = cm

        self.tp = self.cm.diagonal()
        self.fp = sum(self.cm) - self.tp
        self.fn = sum(np.transpose(self.cm)) - self.tp
        self.tn = sum(sum(self.cm)) - (self.tp + self.fp + self.fn)

        self.n = len(self.tp)
        if classes != []:
            self.classes = classes
        else:
            self.classes = [f'Class {i}' for i in range(self.n)]
        self.num_classes = len(self.classes)

        self.sum_tp = sum(self.tp)
        self.sum_fp = sum(self.fp)
        self.sum_fn = sum(self.fn)
        self.sum_tn = sum(self.tn)

        self.accuracies = (self.tp + self.tn)/(self.tp + self.tn + self.fp
            + self.fn + 0.000000001) * 1.0
        self.accuracy_M = sum(self.accuracies)/self.n * 1.0
        self.accuracy_m = (self.sum_tp + self.sum_tn)/(self.sum_tp
            + self.sum_tn + self.sum_fp + self.sum_fn + 0.000000001) * 1.0
        self.accuracy = sum(self.tp)/(sum(sum(self.cm)) + 0.000000001)*1.0

        self.error_rates = (self.fp + self.fn)/(self.tp + self.tn + self.fp
            + self.fn + 0.000000001)*1.0
        self.error_rate_M = sum(self.error_rates)/self.n * 1.0
        self.error_rate_m = (self.sum_fp + self.sum_fn)/(self.sum_tp
            + self.sum_fp + self.sum_fn + self.sum_tn + 0.000000001) * 1.0

        self.precisions = (self.tp)/(self.tp + self.fp + 0.000000001) * 1.0
        self.precision_M = sum(self.precisions)/self.n*1.0
        self.precision_m = self.sum_tp/(self.sum_tp + self.sum_fp
            + 0.000000001) * 1.0

        self.recalls = (self.tp)/(self.tp + self.fn + 0.000000001) * 1.0
        self.recall_M = sum(self.recalls)/self.n*1.0
        self.recall_m = self.sum_tp/(self.sum_tp + self.sum_fn + 0.000000001)

        self.fscores = (self.f_beta**2 + 1.0)*self.precisions*self.recalls/(
            (self.f_beta**2)*self.precisions + self.recalls + 0.000000001)
        self.fscore_M = (self.f_beta**2+1.0)*self.precision_M*self.recall_M/(
            (self.f_beta**2)*self.precision_M + self.recall_M + 0.000000001)
        self.fscore_m = (self.f_beta**2+1.0)*self.precision_m*self.recall_m/(
            (self.f_beta**2)*self.precision_m + self.recall_m + 0.000000001)

        self.specificity = (self.tn)/(self.tn+self.fp + 0.0000000001)*1.0
        self.specificity_M = sum(self.specificity)/self.n*1.0
        self.specificity_m = self.sum_tn/(
            self.sum_tn + self.sum_fp + 0.00000000001)*1.0

    def get_report(self):
        out = f'{"*" * 79}\n**{" " * 26} CLASSIFICATION REPORT '\
            f'{" " * 26}**\n{"*" * 79}\n'
        out += 'Confusion Matrix (row = true, column = predicted):\n'
        out += str(self.cm) + '\n'
        out += '\nStatistics:\n'
        out += f'{"Classes":10s} {"Accuracy":10s} {"Error":10s} '\
            f'{"Precision":10s} {"Recall":10s} {"Specificity":10s} '\
            f'{"F1-score":10s}\n'
        for i in range(self.n):
            out += f'{self.classes[i]:10s} {self.accuracies[i]:10.3f} '\
                f'{self.error_rates[i]:10.3f} {self.precisions[i]:10.3f} '\
                f'{self.recalls[i]:10.3f} {self.specificity[i]:10.3f} '\
                f'{self.fscores[i]:10.3f}\n'
        out += f'\n{"Macro mean":10s} {self.accuracy_M:10.3f} '\
            f'{self.error_rate_M:10.3f} {self.precision_M:10.3f} '\
            f'{self.recall_M:10.3f} {self.specificity_M:10.3f} '\
            f'{self.fscore_M:10.3f}\n'
        out += f'{"Micro mean":10s} {self.accuracy_m:10.3f} '\
            f'{self.error_rate_m:10.3f} {self.precision_m:10.3f} '\
            f'{self.recall_m:10.3f} {self.specificity_m:10.3f} '\
            f'{self.fscore_m:10.3f}\n'
        out += f'{"Accuracy*":10s} {self.accuracy:10.3f}\n'
        return out

    def save_report(self, message='Report'):
        out = f'{"*" * 79}\n**{" " * 26} CLASSIFICATION REPORT '\
            f'{" " * 26}**\n{"*" * 79}\n'
        out += 'Confusion Matrix (row = true, column = predicted):\n'
        out += str(self.cm) + '\n'
        out += '\nStatistics:\n'
        out += f'{"Classes":10s} {"Accuracy":10s} {"Error":10s} '\
            f'{"Precision":10s} {"Recall":10s} {"Specificity":10s} '\
            f'{"F1-score":10s}\n'
        for i in range(self.n):
            out += f'{self.classes[i]:10s} {self.accuracies[i]:10.3f} '\
                f'{self.error_rates[i]:10.3f} {self.precisions[i]:10.3f} '\
                f'{self.recalls[i]:10.3f} {self.specificity[i]:10.3f} '\
                f'{self.fscores[i]:10.3f}\n'
        out += f'\n{"Macro mean":10s} {self.accuracy_M:10.3f} '\
            f'{self.error_rate_M:10.3f} {self.precision_M:10.3f} '\
            f'{self.recall_M:10.3f} {self.specificity_M:10.3f} '\
            f'{self.fscore_M:10.3f}\n'
        out += f'{"Micro mean":10s} {self.accuracy_m:10.3f} '\
            f'{self.error_rate_m:10.3f} {self.precision_m:10.3f} '\
            f'{self.recall_m:10.3f} {self.specificity_m:10.3f} '\
            f'{self.fscore_m:10.3f}\n'
        out += f'{"Accuracy*":10s} {self.accuracy:10.3f}\n'
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'{self.output_dir}/PR_{self.filename_prefix}_'\
                    f'{time_str}.txt', 'w+') as f:
            f.write(out)

    def plot_confusion_matrix(self,
                                normalize: bool=False,
                                title: str='Confusion Matrix',
                                cmap=plt.cm.Blues,
                                cm: List[List[int]]=[]):
        if cm==[]: cm = np.array([[i for i in j] for j in self.cm])
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        data = [[self.classes[i], self.classes[j], cm[i,j]] for i in range(self.num_classes) for j in range(self.num_classes)]
        df = pd.DataFrame(data, columns=['True','Predicted','Amount'])
        df = df.pivot('True','Predicted','Amount')

        plt.figure(figsize=(10,10))
        plt.subplots_adjust(left=0.19, bottom=0.20, right=0.98, top=0.88)
        cm = sns.heatmap(df, annot=True, cmap=plt.cm.Blues, fmt='d')
        cm.set_title('Superfamily Classification')
        cm.set_xticklabels(cm.get_xticklabels(), rotation=45, horizontalalignment='right')
        cm.set_yticklabels(cm.get_yticklabels(), rotation=45, horizontalalignment='right')

        plt.show()
        plt.clf()
        plt.close('all')

    def save_confusion_matrix(self, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        cm = np.array([[i for i in j] for j in self.cm])
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        data = [[self.classes[i], self.classes[j], cm[i,j]] for i in range(self.num_classes) for j in range(self.num_classes)]
        df = pd.DataFrame(data, columns=['True','Predicted','Amount'])
        df = df.pivot('True','Predicted','Amount')

        plt.figure(figsize=(10,10))
        plt.subplots_adjust(left=0.19, bottom=0.20, right=0.98, top=0.88)
        cm = sns.heatmap(df, annot=True, cmap=plt.cm.Blues, fmt='d')
        cm.set_title('Superfamily Classification')
        cm.set_xticklabels(cm.get_xticklabels(), rotation=45, horizontalalignment='right')
        cm.set_yticklabels(cm.get_yticklabels(), rotation=45, horizontalalignment='right')

        plt.savefig(self.output_dir+'/CM_'+self.filename_prefix+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.png')
        plt.clf()
        plt.close('all')

    def plot_learning_curve(self,accuracies,acc=0):
        plt.figure()
        if(acc==0):
            plt.plot([e[0] for e in accuracies],[e[1] for e in accuracies])
        elif(acc==1):
            plt.plot([e[0] for e in accuracies],[e[2] for e in accuracies])
        elif(acc==2):
            plt.plot([e[0] for e in accuracies],[e[3] for e in accuracies])
        plt.show()
        plt.clf()
        plt.close('all')

    def save_learning_curve(self,accuracies,title,acc_type=0):
        acc_type_axis_titles = ['Accuracy(micro)','Accuracy(macro)','Accuracy']
        plt.figure()
        if(acc_type==0):
            plt.plot([e[0] for e in accuracies],[e[1] for e in accuracies])
        elif(acc_type==1):
            plt.plot([e[0] for e in accuracies],[e[2] for e in accuracies])
        elif(acc_type==2):
            plt.plot([e[0] for e in accuracies],[e[3] for e in accuracies])
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(acc_type_axis_titles[acc_type])
        plt.savefig(self.output_dir+'/LC_'+self.filename_prefix+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.png')
        plt.clf()
        plt.close('all')


class CNN_model(object):
    def __init__(self,
                 num_classes: int,
                 classes: List[str],
                 architecture: List[str],
                 activation_functions: List[str],
                 widths: List[int],
                 strides: List[int],
                 dilations: List[int],
                 feature_maps: List[int],
                 vocab_size: int,
                 max_len: int,
                 l2_reg_lambda: float = 0.001,
                 dropout_rate: float = 0.5):
        # initializes weights and biases
        architecture.append('pred')
        activation_functions.append('pred')
        flatten_shape = calculate_flatten_shape(architecture, widths,
                                                feature_maps, max_len)

        # instatiate constants
        tf.constant(num_classes, name='num_classes')
        tf.constant(classes, name='classes')
        tf.constant(architecture, name='architecture')
        tf.constant(activation_functions, name='activation_functions')
        tf.constant(widths, name='widths')
        tf.constant(strides, name='strides')
        tf.constant(dilations, name='dilations')
        tf.constant(feature_maps, name='feature_maps')
        tf.constant(vocab_size, name='vocab_size')
        tf.constant(max_len, name='max_len')
        tf.constant(l2_reg_lambda, name='l2')

        self.is_training = tf.compat.v1.placeholder(tf.bool,
                                                    name='is_training')

        self.dropout_rate = tf.compat.v1.placeholder_with_default(
            dropout_rate,
            shape=(),
            name='dropout_rate'
        )

        self.x_input = tf.compat.v1.placeholder(tf.float32,
                                                [None, max_len, vocab_size, 1],
                                                name="x_input")

        self.y_input = tf.compat.v1.placeholder(tf.float32,
                                                [None, num_classes],
                                                name="y_input")
        # self.dropout = tf.compat.v1.placeholder(tf.float32, name="dropout")

        self.W, self.B = self.create_learnable_params(architecture,
                                                      widths, feature_maps, vocab_size, num_classes, flatten_shape)
        self.layers = self.create_layers(architecture, widths, strides,
                                         dilations, activation_functions, flatten_shape)

        # losses
        loss_sum = 0
        for b in self.B:
            loss_sum += tf.nn.l2_loss(self.B[b])
        for w in self.W:
            loss_sum += tf.nn.l2_loss(self.W[w])

        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.layers['outputs'],
            labels=self.y_input)

        self.loss = tf.reduce_mean(losses) + (l2_reg_lambda * loss_sum)

        # predictions
        self.labels = tf.argmax(self.y_input, 1)
        self.correct_predictions = tf.equal(self.layers['pred'], self.labels)
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_predictions, "float"),
            name="accuracy")

    def create_learnable_params(self,
                                architecture: List[str],
                                widths: List[int],
                                feature_maps: List[int],
                                vocab_size: int,
                                num_classes: int,
                                flatten_shape: int):
        W = dict()
        B = dict()
        conv_counter = 0
        fc_counter = 0

        for i, layer in enumerate(architecture):
            if layer == 'conv':
                height = vocab_size if conv_counter == 0 else 1
                prev_layer_dim = 1 if conv_counter == 0 else feature_maps[conv_counter - 1]
                W['conv' + str(i)] = tf.Variable(
                    tf.random.truncated_normal([widths[i], height, prev_layer_dim, feature_maps[conv_counter]],
                                               stddev=0.1, dtype=tf.float32))
                B['conv' + str(i)] = tf.Variable(
                    tf.random.truncated_normal([feature_maps[conv_counter]], stddev=0.1, dtype=tf.float32))
                conv_counter += 1
            elif layer == 'fc':
                height = flatten_shape if fc_counter == 0 else widths[i - 1]
                W['fc' + str(i)] = tf.Variable(tf.random.truncated_normal([height, widths[i]], stddev=0.1))
                B['fc' + str(i)] = tf.Variable(tf.random.truncated_normal([widths[i]], stddev=0.1))
                fc_counter += 1
            elif layer == 'pred':
                W['pred'] = tf.Variable(tf.random.truncated_normal([widths[i - 1], num_classes], stddev=0.1))
                B['pred'] = tf.Variable(tf.random.truncated_normal([num_classes], stddev=0.1))
        return W, B

    def create_layers(self, architectures, widths, strides, dilations, activation_functions, flatten_shape):
        layers = dict()
        conv_counter = 0
        fc_counter = 0
        prev_layer = self.x_input
        for i, layer in enumerate(architectures):
            if layer == 'conv':
                layers['conv' + str(i)] = tf.nn.conv2d(prev_layer, self.W['conv' + str(i)],
                                                       strides=[strides[i], 1, 1, 1],
                                                       dilations=[1, dilations[conv_counter], 1, 1],
                                                       padding='SAME' if i != 0 else 'VALID')
                if activation_functions[i] == 'relu':
                    layers['relu' + str(i)] = tf.nn.relu(
                        tf.nn.bias_add(layers['conv' + str(i)], self.B['conv' + str(i)]))
                    prev_layer = layers['relu' + str(i)]
                elif activation_functions[i] == 'tanh':
                    layers['tanh' + str(i)] = tf.nn.tanh(
                        tf.nn.bias_add(layers['conv' + str(i)], self.B['conv' + str(i)]))
                    prev_layer = layers['tanh' + str(i)]
                elif activation_functions[i] == 'sigmoid':
                    layers['sigmoid' + str(i)] = tf.nn.sigmoid(
                        tf.nn.bias_add(layers['conv' + str(i)], self.B['conv' + str(i)]))
                    prev_layer = layers['sigmoid' + str(i)]
                elif activation_functions[i] == 'leaky_relu':
                    layers['leaky_relu' + str(i)] = tf.nn.leaky_relu(
                        tf.nn.bias_add(layers['conv' + str(i)], self.B['conv' + str(i)]))
                    prev_layer = layers['leaky_relu' + str(i)]
                elif activation_functions[i] == 'elu':
                    layers['elu' + str(i)] = tf.nn.elu(tf.nn.bias_add(layers['conv' + str(i)], self.B['conv' + str(i)]))
                    prev_layer = layers['elu' + str(i)]
                conv_counter += 1
                # prev_layer = tf.nn.dropout(prev_layer,rate=self.dropout)
            elif layer == 'pool':
                if activation_functions[i] == 'avg':
                    layers['pool' + str(i)] = tf.nn.avg_pool2d(prev_layer, ksize=[1, widths[i], 1, 1],
                                                               strides=[1, strides[i], 1, 1], padding='SAME')
                elif activation_functions[i] == 'max':
                    layers['pool' + str(i)] = tf.nn.max_pool(prev_layer, ksize=[1, widths[i], 1, 1],
                                                             strides=[1, strides[i], 1, 1], padding='SAME')
                prev_layer = layers['pool' + str(i)]
            elif layer == 'fc':
                if fc_counter == 0:
                    layers['flatten'] = tf.reshape(prev_layer, [-1, flatten_shape])
                    prev_layer = layers['flatten']
                if activation_functions[i] == 'relu':
                    layers['relu' + str(i)] = tf.nn.relu(
                        tf.matmul(prev_layer, self.W['fc' + str(i)]) + self.B['fc' + str(i)])
                    prev_layer = layers['relu' + str(i)]
                elif activation_functions[i] == 'tanh':
                    layers['tanh' + str(i)] = tf.nn.tanh(
                        tf.matmul(prev_layer, self.W['fc' + str(i)]) + self.B['fc' + str(i)])
                    prev_layer = layers['tanh' + str(i)]
                elif activation_functions[i] == 'sigmoid':
                    layers['sigmoid' + str(i)] = tf.nn.sigmoid(
                        tf.matmul(prev_layer, self.W['fc' + str(i)]) + self.B['fc' + str(i)])
                    prev_layer = layers['sigmoid' + str(i)]
                elif activation_functions[i] == 'leaky_relu':
                    layers['leaky_relu' + str(i)] = tf.nn.leaky_relu(
                        tf.matmul(prev_layer, self.W['fc' + str(i)]) + self.B['fc' + str(i)])
                    prev_layer = layers['leaky_relu' + str(i)]
                elif activation_functions[i] == 'elu':
                    layers['elu' + str(i)] = tf.nn.elu(
                        tf.matmul(prev_layer, self.W['fc' + str(i)]) + self.B['fc' + str(i)])
                    prev_layer = layers['elu' + str(i)]
                prev_layer = tf.cond(
                    self.is_training,
                    lambda: tf.nn.dropout(prev_layer, rate=self.dropout_rate),
                    lambda: prev_layer
                )
                fc_counter += 1
            elif layer == 'pred':
                layers['outputs'] = tf.nn.bias_add(tf.matmul(prev_layer, self.W['pred']), self.B['pred'],
                                                   name='outputs')
                layers['scores'] = tf.nn.softmax(layers['outputs'], name='scores')
                layers['pred'] = tf.argmax(layers['scores'], 1, name='prediction')
        return layers

    def print_layers_shape(self):
        print('*' * 79 + '\n**' + ' ' * 32 + ' RUN  INFO ' + ' ' * 32 + '**\n' + '*' * 79)
        print('W:')
        for w in self.W:
            print('\t' + w + '\t' + str(self.W[w]))
        print('B:')
        for b in self.B:
            print('\t' + b + '\t' + str(self.B[b]))

        print('*' * 79 + '\n**' + ' ' * 33 + ' TENSORS ' + ' ' * 33 + '**\n' + '*' * 79)
        for layer in self.layers:
            print(str(layer))

# Superfamily dict
superf_dict = {'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
               'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
               'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
               'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
               'KOLOBOK': 28, 'ACADEM-1': 29}

def calculate_flatten_shape(architecture, widths, feature_maps, max_len):
    last_shape = max_len - widths[0] + 1
    for i, layer in enumerate(architecture):
        if layer == 'pool':
            last_shape = int(np.ceil(last_shape / widths[i]))
    return last_shape * feature_maps[-1]


def get_label_data(data_file):
    labels = []
    for record in SeqIO.parse(data_file, "fasta"):
        classification = record.id.split("#")[1].split(" ")[0]
        labels.append(superf_dict[classification])
    return np.asarray(labels)


def print_model(architecture, functions, widths, strides, feature_maps, max_len):
    print('*' * 79 + '\n**' + ' ' * 34 + ' MODEL ' + ' ' * 34 + '**\n' + '*' * 79)
    # print('%20s %s' % ('Classes:',', '.join(t for t in classes)))
    print('%20s %s' % ('Classes:', ', '.join(str(t) for t in classes)))
    print('%20s %d' % ('Max length:',max_len))
    print('%20s %s' % ('Architecture:',''.join('%-8s' % t for t in architecture)))
    print('%20s %s' % ('Functions:',''.join('%-8s' % t for t in functions)))
    print('%20s %s' % ('Widths:',''.join('%-8s' % t for t in widths)))
    print('%20s %s' % ('Strides:',''.join('%-8s' % t for t in strides)))
    feature_maps_string = '%20s ' % 'Feature maps:'
    j = 0
    for i, layer in enumerate(architecture):
        if layer == 'conv':
            feature_maps_string += '%-8s' % str(feature_maps[j])
            j += 1
        else:
            feature_maps_string += '%-8s' % '-'
    print(feature_maps_string)


def get_optimizer(optimizer_option, learning_rate):
    if optimizer_option == 'ADAM':
        return tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer_option == 'ADADELTA':
        return tf.compat.v1.train.AdadeltaOptimizer(learning_rate=learning_rate)
    elif optimizer_option == 'ADAGRAD':
        return tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate)
    elif optimizer_option == 'FTRL':
        return tf.compat.v1.train.FtrlOptimizer(learning_rate=learning_rate)
    elif optimizer_option == 'RMSPROP':
        return tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif optimizer_option == 'GRAD_DESC':
        return tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)


def data_handler(fasta,  max_len=0, mode="T"):
    x = []
    y = []

    with open(fasta,'r') as f:
        for l in f.readlines():
            if l[0] == '>':
                if x != []:
                    x[-1] = np.array([1 if c=='A' else 2 if c=='C' else 3 if c=='G' else 4 if c=='T' else 5 for c in x[-1]],dtype=np.uint8)
                    if mode == "T" and len(x[-1]) > max_len:
                        max_len = len(x[-1])
                x.append('')
            else:
                x[-1] += l.upper().strip()
        x[-1] = np.array([1 if c=='A' else 2 if c=='C' else 3 if c=='G' else 4 if c=='T' else 5 for c in x[-1]],dtype=np.uint8)
        if  mode == "T" and len(x[-1]) > max_len:
            max_len = len(x[-1])
    
    # self.x_train = np.array([np.pad(self.x_train[i],(0,self.max_len-len(self.x_train[i])),'constant',constant_values=(0,0)) for i in range(self.train_size)]) # pads with zero the sequences with length different from max_len
    amount_data = len(x)
    x = np.array([
        np.pad(seq[:max_len], (0, max_len - len(seq[:max_len])), mode='constant') for seq in x[:amount_data]
    ]) # pads with zero the sequences with length different from max_len

    return np.array([e for e in x]), max_len


def warm_start_from_savedmodel(sess, based_dir, import_scope="BASE"):
    if not based_dir or not tf.io.gfile.exists(os.path.join(based_dir, "saved_model.pb")):
        raise ValueError(f"SavedModel no válido en: {based_dir}")

    print(f"[INFO] Loading SavedModel for warm start from: {based_dir}")

    tf.compat.v1.saved_model.loader.load(
        sess,
        [tf.saved_model.SERVING],
        based_dir,
        import_scope=import_scope
    )

    dst_vars = {v.name.split(":")[0]: v for v in tf.compat.v1.global_variables()}

    src_vars = {}
    for v in tf.compat.v1.global_variables():
        name = v.name.split(":")[0]
        if name.startswith(import_scope + "/"):
            src_vars[name[len(import_scope) + 1:]] = v  # quitamos "BASE/" del nombre

    assign_ops = []
    matched, skipped = 0, 0

    for name, dst in dst_vars.items():
        if name.startswith(import_scope + "/"):
            continue

        src = src_vars.get(name)
        if src is None:
            skipped += 1
            continue

        if dst.shape.as_list() != src.shape.as_list():
            skipped += 1
            continue

        assign_ops.append(tf.compat.v1.assign(dst, src))
        matched += 1

    if matched == 0:
        raise RuntimeError(
            "No se pudo emparejar ninguna variable entre el modelo base y el actual. "
            "Probablemente cambiaron nombres o arquitectura."
        )

    sess.run(assign_ops)
    print(f"[INFO] Warm start done. Copied {matched} variables. Skipped {skipped}.")


def train_evaluate(x_train, y_train, x_test, y_test, vocab_size, max_len, classes, num_classes, 
                   architecture, activation_functions, widths, strides, dilations, feature_maps, 
                   optimizer, l2, train_batch_size, test_batch_size, epochs, dropout=0.5, 
                   output_file='Models/'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                   print_results = True, save_model = True, based_weights = ""):
    out = ''
    train_length = len(y_train)
    test_length = len(y_test)
    with tf.compat.v1.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            cnn = CNN_model(
                num_classes,
                classes,
                architecture,
                activation_functions,
                widths,
                strides,
                dilations,
                feature_maps,
                vocab_size,
                max_len,
                l2,
                dropout
            )

            global_step = tf.compat.v1.Variable(0, name="global_step", trainable=False)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            pre_x = tf.compat.v1.placeholder(tf.uint8,[None, max_len], name='pre_x')
            pre_y = tf.compat.v1.placeholder(tf.uint8,[None], name='pre_y')
            one_hot_x = tf.compat.v1.one_hot(pre_x, vocab_size, dtype=tf.float32, name='one_hot_x')
            one_hot_y = tf.compat.v1.one_hot(pre_y, num_classes, dtype=tf.float32, name='one_hot_y')

            accuracies = []
            training_time = 0
            test_times = []
            best_result = [0, [], []]
            sess.run([
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.local_variables_initializer()
            ])

            if based_weights:
                warm_start_from_savedmodel(sess, based_weights, import_scope="BASE")

            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.x_input: x_batch,
                    cnn.y_input: y_batch,
                    cnn.is_training: True
                }
                _, step = sess.run([train_op, global_step], feed_dict)

            def eval_step(x_batch, y_batch):
                feed_dict = {
                    cnn.x_input: x_batch,
                    cnn.y_input: y_batch,
                    cnn.is_training: False
                }
                predictions, scores = sess.run([cnn.layers['pred'], cnn.layers['scores']], feed_dict)
                return predictions, scores
            
            def evaluate(epoch, test_len, batch_size, x_test, y_test):
                predictions = np.array([], dtype=np.uint8)
                scores = None
                for i in range(0, test_len, batch_size):
                    x_batch = x_test[i : i + batch_size]
                    y_batch = y_test[i : i + batch_size]
                    pre_xo = sess.run(one_hot_x, feed_dict={pre_x:x_batch})
                    x_batch = pre_xo.reshape(x_batch.shape[0], max_len, vocab_size, 1)
                    y_batch = sess.run(one_hot_y, feed_dict={pre_y:y_batch})
                    preds, scr = eval_step(x_batch, y_batch)
                    predictions = np.concatenate([predictions, preds])
                    if scores is not None:
                        scores = np.concatenate([scores, scr])
                    else:
                        scores = scr
                m = Metric(y_test, predictions, classes=classes)
                accuracies.append([epoch, m.accuracy_M, m.accuracy_m, m.accuracy])
                return predictions, scores

            #TRAIN
            training_time = time.time()
            for epoch in range(epochs):
                print(f'Epoch: {epoch + 1} / {epochs}')
                for batch in range(0, train_length, train_batch_size):
                    x_batch = x_train[batch : batch + train_batch_size]
                    y_batch = y_train[batch : batch + train_batch_size]
                    pre_xo = sess.run(one_hot_x, feed_dict={pre_x:x_batch})
                    x_batch = pre_xo.reshape(x_batch.shape[0], max_len, vocab_size, 1)
                    y_batch = sess.run(one_hot_y, feed_dict={pre_y:y_batch})
                    train_step(x_batch, y_batch)
                    current_step = tf.compat.v1.train.global_step(sess, global_step)
                shuffle_indices = np.random.permutation(range(train_length))
                x_train = x_train[shuffle_indices]
                y_train = y_train[shuffle_indices]
                test_times.append(time.time())
                predictions, scores = evaluate(epoch, test_length, test_batch_size, x_test, y_test)
                test_times[-1] = time.time() - test_times[-1]
                if accuracies[-1][1] > best_result[0]:
                    best_result = [accuracies[-1][1], np.copy(predictions), np.copy(scores)]
                time_str = datetime.datetime.now().isoformat()
                out += time_str+': '+str(accuracies[-1]) + '\n'
                if print_results: print(time_str+': '+str(accuracies[-1]))
            
            training_time = time.time() - training_time

            output_path = output_file + '_' + str(len(accuracies))
            if os.path.exists(output_path):
                print(f"Cleaning {output_path} ...")
                shutil.rmtree(output_path)

            probs = tf.identity(cnn.layers['scores'], name='probabilities')
            if save_model:
                print("Saving Model")
                tf.compat.v1.saved_model.simple_save(
                    sess,
                    output_path,
                    inputs={
                        'x_input': cnn.x_input,
                        'is_training': cnn.is_training
                    },
                    outputs={'prediction': cnn.layers['pred'],
                             'probabilities': probs}
                )
            
    return y_test, np.array(predictions[1], dtype=np.uint8), accuracies, best_result, training_time, out