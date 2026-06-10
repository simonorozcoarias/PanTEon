# -*- coding: utf-8 -*-
import pandas as pd
from Bio import SeqIO
import math
import seaborn as sn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import Adam
from typing import List, Tuple, Dict

try:
    from hierarchicalsoftmax import (
        SoftmaxNode,
        HierarchicalSoftmaxLazyLinear,
        HierarchicalSoftmaxLoss,
        greedy_accuracy,
        greedy_f1_score,
        greedy_predictions
    )
except Exception as e:
    raise SystemExit("[ERROR] Package 'hierarchicalsoftmax' couldn't be loaded. Install it with:  pip install hierarchicalsoftmax\nDetails: %s" % e)


class TerrierNet(nn.Module):
    def __init__(self, root: SoftmaxNode,
                 vocab_size: int, emb_dim: int = 18, n_layers: int = 4, k: int = 7,
                 growth: float = 1.96, first_channels: int = 64,
                 dropout: float = 0.248, penult: int = 1953):
        super().__init__()
        self.root = root
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        chans = [int(round(first_channels * (growth ** i))) for i in range(n_layers)]
        blocks, in_c = [], emb_dim
        for out_c in chans:
            blocks.append(nn.Conv1d(in_c, out_c, kernel_size=k, padding=k // 2))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Dropout(dropout))
            blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_c = out_c
        self.conv = nn.Sequential(*blocks)

        self.penultimate = nn.Linear(chans[-1], penult)
        self.hsoftmax = HierarchicalSoftmaxLazyLinear(root=root)

    def forward(self, x):
        x = self.embedding(x)        # (B, L, E)
        h = x.transpose(1, 2)        # (B, E, L)
        h = self.conv(h)             # (B, C, L')
        h = h.mean(dim=2)            # GAP
        z = torch.relu(self.penultimate(h))
        logits = self.hsoftmax(z)    # (B, root.layer_size), without softmax (raw logits)
        return logits


class TransposonDataset(Dataset):
    def __init__(self, X: np.ndarray, y_node_ids: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.int64))
        self.y = torch.from_numpy(y_node_ids.astype(np.int64))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]


class InferenceDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X.astype(np.int64))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i]

# Superfamily dict
superf_dict = {'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
               'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
               'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
               'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
               'KOLOBOK': 28, 'ACADEM-1': 29}
order_dict = {'LTR': 0, 'LINE': 1, 'DIRS': 2, 'PLE': 3, 'SINE': 4, 'HELITRON': 5, 'CRYPTON': 6, 'TIR': 7 }

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DNA_ALPHABET = "ACGTN"
BASE2IDX = {b: i+1 for i,b in enumerate(DNA_ALPHABET)}  # 0=PAD

def clean_seq(seq):
    s = str(seq).upper()
    return "".join(ch if ch in DNA_ALPHABET else "N" for ch in s)


def encode_sequence(seq: str, max_len: int) -> np.ndarray:
    arr = np.zeros(max_len, dtype=np.int64)
    L = min(len(seq), max_len)
    for i in range(L):
        arr[i] = BASE2IDX.get(seq[i], 0)
    return arr


def build_hierarchy(order_labels: List[str], superf_labels: List[str], phi: float = 1.02):
    root = SoftmaxNode("ROOT")
    root.alpha = 1.0

    order_names = sorted(set(order_labels))
    order_nodes: Dict[str, SoftmaxNode] = {name: SoftmaxNode(f"ORDER::{name}", parent=root) for name in order_names}
    for node in order_nodes.values():
        node.alpha = phi

    sf_pairs = sorted(set(zip(order_labels, superf_labels)))
    superf_nodes: Dict[Tuple[str, str], SoftmaxNode] = {}
    for o, s in sf_pairs:
        o_node = order_nodes[o]
        s_node = SoftmaxNode(f"SUPERF::{o}::{s}", parent=o_node)
        superf_nodes[(o, s)] = s_node

    root.set_indexes()

    order_id_map = {name: order_nodes[name] for name in order_names}
    superf_id_map = {(o, s): superf_nodes[(o, s)] for (o, s) in sf_pairs}

    return root, order_id_map, superf_id_map


def targets_to_node_ids(root: SoftmaxNode, order_labels: List[str], superf_labels: List[str],
                        order_node_map: Dict[str, SoftmaxNode],
                        superf_node_map: Dict[Tuple[str, str], SoftmaxNode]) -> np.ndarray:
    nodes = []
    for o, s in zip(order_labels, superf_labels):
        nodes.append(superf_node_map[(o, s)])
    ids = root.get_node_ids(nodes)
    return np.asarray(ids, dtype=np.int64)


def build_superf_id_maps(root, superf_node_map):
    pair2id = {}
    id2pair = {}
    for (order, superf), node in superf_node_map.items():
        node_id = int(root.get_node_ids([node])[0])
        pair2id[(order, superf)] = node_id
        id2pair[node_id] = (order, superf)
    return pair2id, id2pair


def load_data(TE_lib, max_len, mode="T"):
    seqs = []
    classifications_superf = []
    classifications_order = []

    for te in SeqIO.parse(TE_lib, "fasta"):
        seqs.append(encode_sequence(clean_seq(te.seq), max_len))
        if mode == "T":
            superfamily = te.id.split("#")[1].split(" ")[0].split("/")[-1]
            order = te.id.split("#")[1].split(" ")[0].split("/")[-2]
            classifications_superf.append(superf_dict[superfamily])
            classifications_order.append(order_dict[order])
        elif mode == "P":
            classifications_order.append(te.id)
        else:
            return None, None, None

    X = np.asarray(seqs)
    Y_order = np.asarray(classifications_order)
    Y_superf = np.asarray(classifications_superf)
    return X, Y_order, Y_superf


def run_experiment(model: nn.Module, root: SoftmaxNode,
          train_loader: DataLoader, val_loader: DataLoader, device,
          epochs=100, max_lr=1e-3, patience=10, weight_decay=0.0) -> Tuple[nn.Module, List[Dict]]:
    model = model.to(device)
    opt = Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = OneCycleLR(opt, max_lr=max_lr, epochs=epochs,
                           steps_per_epoch=len(train_loader), anneal_strategy="cos", pct_start=0.3)
    hs_loss = HierarchicalSoftmaxLoss(root)

    best_val = math.inf
    best_state = None
    hist = []
    no_imp = 0

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = hs_loss(logits, yb)
            loss.backward()
            opt.step()
            scheduler.step()
            run_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
        tr_loss = run_loss / max(1, seen)

        model.eval()
        with torch.no_grad():
            val_logits_list, val_targets_list = [], []
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                val_logits_list.append(logits.cpu())
                val_targets_list.append(yb)
            val_logits = torch.cat(val_logits_list, dim=0)
            val_targets = torch.cat(val_targets_list, dim=0)
            val_metrics = evaluate(val_logits, val_targets, root)
            val_loss = float(HierarchicalSoftmaxLoss(root)(val_logits, val_targets).item())

        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, **val_metrics}
        hist.append(row)
        print(f"[INFO] [Epoch {epoch}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f} ",
              f"F1_superf={val_metrics['superf_f1_weighted']:.4f} F1_order={val_metrics['order_f1_weighted']:.4f}")

        if val_loss < best_val - 1e-9:
            best_val, best_state, no_imp = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"[INFO] Early stopping @ epoch {epoch}. best val_loss={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, hist


def evaluate(logits: torch.Tensor, targets: torch.Tensor, root: SoftmaxNode) -> Dict[str, float]:
    acc_superf = greedy_accuracy(logits, targets, root)
    f1_superf = greedy_f1_score(logits, targets, root, average="weighted")

    acc_order = greedy_accuracy(logits, targets, root, max_depth=1)
    f1_order = greedy_f1_score(logits, targets, root, average="weighted", max_depth=1)

    return {
        "superf_acc": float(acc_superf),
        "superf_f1_weighted": float(f1_superf),
        "order_acc": float(acc_order),
        "order_f1_weighted": float(f1_order),
    }


def to_order_label(node_name: str) -> str:
    parts = node_name.split("::")
    if len(parts) >= 3:
        return parts[1]
    if len(parts) == 2 and parts[0] == "ORDER":
        return parts[1]
    return node_name