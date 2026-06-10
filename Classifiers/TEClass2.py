# -*- coding: utf-8 -*-
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, \
    classification_report, precision_recall_fscore_support
from Bio import SeqIO
import numpy as np
import os, sys
os.environ["KERAS_BACKEND"] = "tensorflow"
import tf_keras as keras
sys.modules["keras"] = keras
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
import tokenizers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from torch.utils.data import Dataset
import torch
import random
import transformers
from transformers import Trainer, TrainingArguments, LongformerConfig
from transformers import LongformerForSequenceClassification, AutoModelForSequenceClassification
from math import pi
from torchvision import transforms
torch.autograd.set_detect_anomaly(True)
global model


#holds the likeability of specific SNPs for each AGTC/AGTC mapping
snp_matrix = [[0,1,1,1],    #A
              [1,0,1,1],    #G
              [1,1,0,1],    #T
              [1,1,1,0],    #C
              [1,1,1,1]]    #ambuigity N will be mapped to AGTC

#define the mappings as literal dor each base
base2key_map = {'A':0 , 'G': 1, 'T':2, 'C':3, 'N':4}
key2base_map = {0:'A' , 1:'G', 2:'T', 3:'C', 'N':4}

#define the complement for each base
complement_map = {ord('G'):'C', ord('T'):'A', ord('C'):'G', ord('A'):'T', ord('N'):'N'}
complement_map_int = {ord('0'):'A', ord('1'):'G', ord('2'):'T', ord('3'):'C', ord('4'):'N'}

superf_dict = {'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
                       'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
                       'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
                       'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
                       'KOLOBOK': 28, 'ACADEM-1': 29}

"""superf_dict = {'negative/negative': 0, 'positive/positive': 1}"""


class TransposonDataset(Dataset):

    def __init__(self, data, datadict_, tokenizer, embedd_larger_seq=True, train=False):
        # save original sequences
        self.seqs = [seq[0] for seq in data]

        # save sequence ids
        self.ids = [id[1] for id in data]

        # save kmer embeddings and embedding_with when kmers are dilated
        embeddings = [seq2kmer(seq[0], embedd_larger_seq) for seq in data]
        kmers, embed_w = [kmer[0] for kmer in embeddings], [kmer[1] for kmer in embeddings]
        self.embed_w = embed_w

        # tokenize kmer into ids and attention masks
        # tokenized_input = [tokenizer.encode(kmer) for kmer in kmers]
        # self.encoded_kmers = [enc_ids.ids for enc_ids in tokenized_input]
        # self.attention_masks = [att_mask.attention_mask for att_mask in tokenized_input]

        self.global_att_tokens = np.array([0, 256, 512])
        self.beta = 1.0

        # save labels
        self.labels = [label[2] for label in data]

        # save additional parameters
        self.tokenizer = tokenizer
        self.train = train

        sample_weight = []
        eps = 0
        if len(datadict_.keys()) > 1: # Do it only for training, no for inference
            for i in range(len(datadict_.keys())):
                sample_weight.append(self.labels.count(i) + eps)

            self.sample_weight = [1 / ((x / sum(sample_weight)) )  for x in sample_weight]  # get inverse of occurence weigths -> large occurences weigth less
        else:
            self.sample_weight = np.ones(1, dtype=np.float32)
        self.sample_weight = torch.tensor(self.sample_weight)

        uniform = torch.tensor(1 / len(datadict_.keys()))
        uniform = uniform.repeat(len(datadict_.keys()))
        self.sample_weight = self.beta * self.sample_weight + (1 - self.beta) * uniform

        print("[INFO] dataset size: ", len(self.labels))

    def getembedding_w(self, idx):
        embed_w = self.embed_w[idx]
        return embed_w

    def getoriginalseq(self, idx):
        return self.seqs[idx], self.ids[idx]

    def getseqids(self):
        return self.ids

    def getkmer(self, seq_index, kmer_pos):
        seq = self.seqs[seq_index]

        if kmer_pos >= len(seq2kmer(seq)[0].split()):
            return '[PAD]'  # attention on padding

        kmer = seq2kmer(seq)[0].split()[kmer_pos]
        return kmer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        seq = self.seqs[idx]

        # kmers = self.encoded_kmers[idx]
        # att_mask = self.attention_masks[idx]

        # Apply augmentation
        # define a composed transformation list which are used for each sequence
        transform = transforms.Compose([normalize(), mask(), compose_([insertion(max_length=7), identity()]),
                                        compose_([deletion(n=1, max_length=7), identity()]),
                                        compose_([repeat(), identity()]),
                                        compose_([complement(), reverse(), reverse_complement(), identity()]),
                                        compose_([add_tail(), remove_tail(), identity()])
                                        ])

        if self.train and random.random() > 1 - 0.55:
            seq = transform(seq)

        kmers = seq2kmer(seq)[0]
        encoded_input = self.tokenizer.encode(kmers)
        kmers = encoded_input.ids
        att_mask = encoded_input.attention_mask

        # generate global attention mask
        global_att_mask = torch.zeros_like(torch.tensor(att_mask), dtype=torch.long)  # , device=att_mask.device)
        global_att_tokens = self.global_att_tokens[self.global_att_tokens <= len(kmers)]
        global_att_mask[global_att_tokens] = 1  # global attentions

        return {"input_ids": kmers, "attention_mask": att_mask, "global_attention_mask": global_att_mask,
                "labels": label}


class normalize(object):
    def __call__(self, seq):
        seq = seq.upper()
        if not all(s in base2key_map for s in seq):
            replace_chars = [char for char in seq if char not in base2key_map.keys()]
            replace_chars = list(dict.fromkeys(replace_chars))
            for c in replace_chars:
                seq = seq.replace(c, 'N')
        return seq


class snp(object):

    def __init__(self, n=1, matrix=snp_matrix):
        self.n = n
        self.snp_matrix = matrix

    def __call__(self, seq):
        for i in range(self.n):
            j = random.randint(0, len(seq) - 1)
            snp_ = base2key_map[seq[j]]
            snp_ = self.snp_matrix[snp_] * np.random.rand(4)
            snp_ = key2base_map[int(snp_.max())]
            seq = seq[:j] + snp_ + seq[j + 1:]
        return seq


class mask(object):

    def __init__(self, n=1, length=5, pos=[0.05, 0.95]):
        self.n = n
        self.length = length
        self.pos = pos

    def __call__(self, seq):
        for i in range(self.n):
            j = int(random.uniform(*self.pos) * len(seq))
            mask_ = 'N' * self.length
            seq = seq[:j] + mask_ + seq[j + self.length:]
        return seq


class insertion(object):

    def __init__(self, min_length=5, max_length=20, pos=[0.05, 0.95], insert_seq=None):
        self.min_length = min_length
        self.max_length = max_length
        self.insert_seq = insert_seq
        self.pos = pos

    def __call__(self, seq):
        j = int(random.uniform(*self.pos) * len(seq))
        if self.insert_seq:
            insert_ = self.insert_seq
        else:
            ##create random sequence
            length = random.randint(self.min_length, self.max_length)
            rand_list = random.choices(range(0, 3), k=length)
            rand_list = "".join(str(e) for e in rand_list)
            insert_ = rand_list.translate(complement_map_int)
        return seq[:j] + insert_ + seq[j:]


class deletion(object):

    def __init__(self, n=1, min_length=5, max_length=20, pos=[0.05, 0.95]):
        self.n = n
        self.min_length = min_length
        self.max_length = max_length
        self.pos = pos

    def __call__(self, seq):
        for i in range(self.n):
            j = int(random.uniform(*self.pos) * len(seq))
            length = random.randint(self.min_length, self.max_length)
            seq = seq[:j] + seq[j + length:]
        return seq


class repeat(object):

    def __init__(self, length=5, min_dist=0, pos=[0.05, 0.95]):
        self.min_dist = min_dist
        self.length = length
        self.pos = pos

    def __call__(self, seq):
        min_dist = (self.min_dist / len(seq))  # to 0.0-1.0 map
        j = int((len(seq) - self.length) * random.uniform(self.pos[0], 1 - self.min_dist))
        repeat_ = seq[j:j + self.length]  # get part seq
        j = int(len(seq) * random.uniform((j + self.length) / len(seq) + self.min_dist, self.pos[1]))
        seq = seq[:j] + repeat_ + seq[j:]
        return seq


class reverse(object):

    def __call__(self, seq):
        return seq[::-1]


class complement(object):

    def __init__(self, complement_map=complement_map):
        self.complement_map = complement_map

    def __call__(self, seq):
        return seq.translate(complement_map)


class reverse_complement(object):

    def __init__(self, complement_map=complement_map):
        self.complement_map = complement_map

    def __call__(self, seq):
        return seq[::-1].translate(complement_map)


class add_tail(object):

    def __init__(self, tail_type='A', length=[5, 20]):
        self.tail_type = tail_type
        self.length = length

    def __call__(self, seq):
        j = random.randint(*self.length)
        return seq + self.tail_type * j


class remove_tail(object):

    def __init__(self, tail_type='A'):
        self.tail_type = tail_type

    def __call__(self, seq):
        while (seq[::-1].find(self.tail_type)) <= 0:
            seq = seq[:-2]
        return seq


class inject_transposons(object):

    def __init__(self, pos=[0.05, 0.95], create_tsd=True, tsd_len=[5, 20]):
        self.pos = pos
        self.create_tsd = create_tsd
        self.tsd_len = tsd_len

    def __call__(self, seq):
        transposon = ''  # get from database
        if self.create_tsd:
            tsd = ''  # create tsd
            transposon = tsd + transposon + tsd
        j = int((len(seq) - len(transposon)) * random.uniform(*self.pos))
        seq = seq[:j] + transposon + seq[j:]

        return seq


class identity(object):

    def __call__(self, seq):
        return seq


class compose_(object):
    def __init__(self, list):
        self.list = list

    def __call__(self, seq):
        return one_of([*self.list])(seq)


class DNAFormer_Trainer(Trainer):

    def __init__(self, sample_weight=[], *args, **kwargs):
        super(DNAFormer_Trainer, self).__init__(*args, **kwargs)
        device = next(self.model.parameters()).device
        self.sample_weight = sample_weight.to(device)
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=self.sample_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if logits.dim() == 3:
            logits = logits[:, 0, :]

        loss = self.loss_fct(logits.float(), labels)

        return (loss, outputs) if return_outputs else loss


class Distributions():

    def __init__(self, dist_type, nsamples, dim):
        self.type = dist_type
        self.nsamples = nsamples * 5
        self.dim = dim

        self.radius = 10  # for circle
        self.expansion = 1.5  # for circle
        self.angle = torch.tensor(2 * pi)  # for cross

    def __call__(self):
        if self.type == 'gauss':
            dist = torch.randn(self.nsamples, self.dim, device='cuda')
        elif self.type == 'circular':
            # random angle and radius
            angle = 2 * pi * torch.randn(self.nsamples, self.dim, device='cuda')
            r = self.radius + torch.randn(self.nsamples, self.dim, device='cuda') * self.expansion

            # coordinates
            x = r * torch.cos(angle)
            y = r * torch.sin(angle)
            dist = torch.concat((x, y), axis=1).reshape(self.nsamples * 2, self.dim)

        elif self.type == 'cross':
            x = torch.randn(self.nsamples, self.ndim, device='cuda')
            y = torch.randn(self.nsamples, self.ndim, device='cuda')

            x = (x * torch.cos(self.angle) * y)
            y = (y * torch.sin(self.angle) * x)
            dist = torch.concat((x, y), axis=1).reshape(self.nsamples * 2, self.dim)

        else:
            raise Exception("No such distribution: ", self.type)
        return dist


class TrainingHistory:
    """Keras like history object for training"""
    def __init__(self, log_history):
        self.history = {
            'loss': [],
            'val_loss': [],
            'f1_m': [],
            'val_f1_m': []
        }
        for entry in log_history:
            if "loss" in entry and "epoch" in entry:
                self.history['loss'].append(entry['loss'])
            if "eval_loss" in entry:
                self.history['val_loss'].append(entry['eval_loss'])
            if "f1" in entry:
                self.history['f1_m'].append(entry['f1'])
            if "eval_f1" in entry:
                self.history['val_f1_m'].append(entry['eval_f1'])


def one_of(transform_list):
    return transform_list[random.randint(0, (len(transform_list)-1))]


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": float(acc), "f1": float(f1), "precision": float(precision), "recall": float(recall)}


def get_model(vocab_file, num_classes):
    global model, model_config

    model_config = LongformerConfig(attention_window=64,
                                    vocab_size=len(vocab_file),
                                    max_position_embeddings=2048,
                                    num_labels=num_classes,
                                    hidden_size=768,
                                    num_hidden_layers=8,
                                    num_attention_heads=8,
                                    intermediate_size=3072,
                                    position_embedding_type='absolute',
                                    problem_type="single_label_classification",
                                    output_attentions=False,
                                    return_dict=True,
                                    pad_token_id=0,
                                    bos_token_id=2,
                                    eos_token_id=3)
    model = LongformerForSequenceClassification(model_config)

    return model


def fasta2dict(datadict_, file_name, mode="T"):
    data = SeqIO.parse(file_name, "fasta")
    for l, entry in enumerate(iter(data)):
        try:
            description = entry.description
            if mode == "T":
                seq_type = description.split(" ")[0].split("#")[1]
                if seq_type in datadict_.keys():
                    datadict_[seq_type].append([str(entry.seq), str(entry.id)])
                else:
                    print(f"[ERROR] Sequence not found in dataset --> {seq_type}")
            elif mode == "P":
                seq_type = "no_class"
                if seq_type in datadict_.keys():
                    datadict_[seq_type].append([str(entry.seq), str(entry.id)])
        except Exception as e:
            raise Exception('Error while loading ', file_name, '\n', e)

    return datadict_


def load_vocab(file_name='data/5mer_vocab'):
    vocab_file = {}
    with open(file_name + '.txt', 'r', encoding="utf-8") as f:
        tokens = f.readlines()
    for i, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab_file[token] = i

    return vocab_file


def create_vocab(k=5):
    import itertools
    list = ["".join(x) for x in itertools.product(["A", "G", "T", "C", "N"], repeat=k)]

    dict_list = []
    dictionary = {}
    dictionary["[PAD]"] = 0
    dictionary["[UNK]"] = 1
    dictionary["[CLS]"] = 2
    dictionary["[SEP]"] = 3
    dictionary["[MSK]"] = 4
    for i, k in enumerate(list):
        dictionary[k] = i + 5

    print(dictionary)
    exit()


def seq2kmer(seq, embedd_larger_seq=True, max_len=1024, k=5):
    # dilate kmers with larger embedding_window 'w'
    if embedd_larger_seq:
        w = min(max(2, int(len(seq) / max_len)), k - 1)  # compute w dynamically up to k steps
    else:
        w = 1
    kmer = [seq[i:i + k] for i in range(0, len(seq) + 1 - k, w)]
    kmer = " ".join(kmer)
    return kmer, w


def split_dataset(dataset, train=0.75, valid=0.15, test=0.1):
    train_len = int(train*len(dataset))
    valid_len = int(valid*len(dataset))
    return dataset[:train_len], dataset[train_len:train_len+valid_len], dataset[train_len+valid_len:]


def dict2dataset(data_, classification_map, normalization=True, mode="T", training=0.75, validation=0.15, test_set=0.1):
    # change to list style
    dataset_train, dataset_valid, dataset_test = [], [], []
    dataset_predict, labels = [], []
    norm = normalize()
    for key in data_.keys():
        dataset_ = []
        for seq in (j for j in data_[key]):

            if mode == "T":
                if normalization:
                    dataset_ += [[norm(seq[0]), seq[1], classification_map[key]]]
                else:
                    dataset_ += [[seq[0], seq[1], classification_map[key]]]

            if mode == "P":
                if normalization:
                    dataset_ += [[norm(seq[0]), seq[1], "no_class"]]
                else:
                    dataset_ += [[seq[0], seq[1], "no_class"]]
                labels.append(seq[1])

        if mode == "T":
            train, valid, test = split_dataset(dataset_,training, validation, test_set)
            dataset_train += list(train)
            dataset_valid += list(valid)
            dataset_test += list(test)
        elif mode == "P":
            dataset_predict += list(dataset_)


    if mode == "T":
        return dataset_train, dataset_valid, dataset_test
    elif mode == "P":
        return dataset_predict, labels
    else:
        return None


def load_data(fasta_path, mode="T", training=0.75, valid=0.15, test=0.1):
    if mode == "T":
        # create empty dict
        te_keywords = list(superf_dict.keys())
        datadict_ = {i: [] for i in te_keywords}

        classification_map = {i: te_keywords.index(i) for i in te_keywords}

        dict = fasta2dict(datadict_, file_name=fasta_path)
        dataset_train, dataset_valid, dataset_test = dict2dataset(dict, classification_map, mode=mode, training=training, validation=valid, test_set=test)

        return dataset_train, dataset_valid, dataset_test, datadict_

    elif mode == "P":
        datadict_ = {"no_class": []}
        dict = fasta2dict(datadict_, file_name=fasta_path, mode=mode)
        dataset_predict, labels = dict2dataset(dict, [], True, mode)
        return dataset_predict, datadict_, labels
    else:
        return None


def tokenizer_fun(PanTEon_dir):
    # initialize kmer tokenizer
    kmer_tokenizer = tokenizers.Tokenizer(
        WordLevel.from_file(f"{PanTEon_dir}/data/5mer_vocab.json", unk_token="[UNK]"))
    kmer_tokenizer.pre_tokenizer = Whitespace()
    kmer_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[PAD]", 0),
            ("[UNK]", 1),
            ("[CLS]", 2),
            ("[SEP]", 3),
            ("[MSK]", 4),
        ],
    )
    max_length = 2046  # take additional "[CLS] $A [SEP]" into account
    kmer_tokenizer.enable_truncation(max_length=max_length)
    kmer_tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_length,
                                  pad_to_multiple_of=64)
    return kmer_tokenizer


def is_main_process():
    return int(os.environ.get("RANK", os.environ.get("PROCESS_RANK", "0"))) == 0


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def setup_device_and_log():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        print(f"[INFO] CUDA Available: {n} GPU(s) -> {names}")
        if "LOCAL_RANK" in os.environ:
            lr = get_local_rank()
            torch.cuda.set_device(lr)
            print(f"[INFO] LOCAL_RANK={lr} -> usando cuda:{lr}")
    else:
        print("[INFO] CUDA not available; CPU will be used")