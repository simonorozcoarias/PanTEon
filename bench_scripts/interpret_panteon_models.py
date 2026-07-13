#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from itertools import product

import tensorflow as tf
from tensorflow.keras.models import load_model

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def integrated_gradients_keras(model, inputs, baseline=None, target_class=None, steps=50):
    is_multi_input = isinstance(inputs, (list, tuple))
    if not is_multi_input:
        inputs = [inputs]

    inputs = [tf.convert_to_tensor(x, dtype=tf.float32) for x in inputs]

    if baseline is None:
        baseline = [tf.zeros_like(x) for x in inputs]
    elif not isinstance(baseline, (list, tuple)):
        baseline = [baseline]

    baseline = [tf.convert_to_tensor(x, dtype=tf.float32) for x in baseline]

    if target_class is None:
        preds = model(inputs if is_multi_input else inputs[0])
        target_class = tf.argmax(preds, axis=1)

    target_class = tf.convert_to_tensor(target_class, dtype=tf.int64)

    total_grads = [tf.zeros_like(x) for x in inputs]

    for alpha in np.linspace(0.0, 1.0, steps):
        alpha = tf.cast(alpha, tf.float32)

        interpolated = [
            b + alpha * (x - b)
            for x, b in zip(inputs, baseline)
        ]

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            preds = model(interpolated if is_multi_input else interpolated[0])
            selected = tf.gather(preds, target_class, batch_dims=1)

        grads = tape.gradient(selected, interpolated)
        total_grads = [
            tg + g
            for tg, g in zip(total_grads, grads)
        ]

    avg_grads = [g / float(steps) for g in total_grads]

    attributions = [
        (x - b) * g
        for x, b, g in zip(inputs, baseline, avg_grads)
    ]

    return [a.numpy() for a in attributions]


def plot_vector_attribution(attributions, out_png, title, feature_names=None, top_k=5):
    attr = np.abs(attributions).mean(axis=0)
    if attr.ndim == 2:
        attr = attr[:, 0]

    attr = attr.reshape(-1)
    
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(attr, color="black", linewidth=1.2)

    # Top-k most important features
    top_idx = np.argsort(attr)[::-1][:top_k]

    for rank, idx in enumerate(top_idx):
        label = (
            feature_names[idx]
            if feature_names is not None
            else f"Feature {idx}"
        )

        ax.scatter(idx, attr[idx], color="red", s=30, zorder=5)

        ax.annotate(
            label,
            xy=(idx, attr[idx]),
            xytext=(0, 18 + rank * 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            arrowprops=dict(
                arrowstyle="-",
                lw=0.8,
                color="gray"
            ),
        )

    ax.set_xlabel("Feature index")
    ax.set_ylabel("Mean absolute attribution")
    ax.set_title("")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_neuralte_feature_attribution(attributions, feature_groups, out_png, title):
    from matplotlib.patches import Patch

    attr = np.abs(attributions).mean(axis=0)
    if attr.ndim == 2:
        attr = attr[:, 0]

    fig, (ax, ax_bar) = plt.subplots(
        2,
        1,
        figsize=(12, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [12, 1], "hspace": 0.05},
    )

    ax.plot(attr, color="black", linewidth=1.2)
    ax.set_ylabel("Mean absolute attribution")
    ax.set_title(title)

    for group_name, start, end, color in feature_groups:
        ax_bar.axvspan(start, end, color=color)

    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Feature index", labelpad=8)
    ax_bar.set_xlim(0, len(attr) - 1)
    ax_bar.set_frame_on(False)

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=group_name)
        for group_name, start, end, color in feature_groups
    ]

    ax_bar.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -1.2),
        ncol=min(len(feature_groups), 4),
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sequence_attribution(attributions, out_png, title):
    attr = np.abs(attributions).sum(axis=-1)
    mean_attr = attr.mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(mean_attr, color="black", linewidth=1.2)

    ax.set_xlim(0, len(mean_attr) - 1)

    ax.set_xticks([150, 450])
    ax.set_xticklabels([
        "5' terminal\n300 bp",
        "3' terminal\n300 bp"
    ])

    ax.set_xlabel("CREATE one-hot sequence representation")
    ax.set_ylabel("Mean absolute attribution")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def build_neuralte_feature_names(internal_kmer_sizes, terminal_kmer_sizes, domain_classes):
    names = []
    groups = []

    kmer_alphabet = "AGCT"

    for k in internal_kmer_sizes:
        for kmer in product(kmer_alphabet, repeat=k):
            names.append(f"internal_{k}mer_{''.join(kmer)}")
            groups.append("Internal k-mers")

    # NeuralTE appends LTR and TIR counts alternately for each k.
    for k in terminal_kmer_sizes:
        kmers = ["".join(x) for x in product(kmer_alphabet, repeat=k)]
        for terminal_type in ("LTR", "TIR"):
            for kmer in kmers:
                names.append(f"{terminal_type}_{k}mer_{kmer}")
                groups.append("Terminal k-mers")

    end_base_order = "ATCG"
    end_positions = [
        "5prime_pos1", "5prime_pos2", "5prime_pos3", "5prime_pos4", "5prime_pos5",
        "3prime_pos_minus4", "3prime_pos_minus3", "3prime_pos_minus2",
        "3prime_pos_minus1", "3prime_last_pos",
    ]
    for position in end_positions:
        for base in end_base_order:
            names.append(f"{position}_{base}")
            groups.append("Sequence ends")

    ordered_domains = [
        name for name, _ in sorted(domain_classes.items(), key=lambda item: item[1])
    ]
    for domain in ordered_domains:
        names.append(f"domain_{domain}")
        groups.append("Domain features")

    return names, groups


def save_feature_group_attributions(attributions, feature_groups, out_csv):
    attr = np.abs(attributions).mean(axis=0)
    if attr.ndim == 2:
        attr = attr[:, 0]

    rows = []
    for group_name, start, end in feature_groups:
        rows.append({
            "feature_group": group_name,
            "start": start,
            "end": end,
            "mean_abs_attribution": float(np.mean(attr[start:end])),
            "sum_abs_attribution": float(np.sum(attr[start:end])),
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def save_neuralte_feature_attributions(attributions, feature_names, feature_group_names, out_csv):
    attr = np.abs(attributions).mean(axis=0).reshape(-1)

    if len(attr) != len(feature_names):
        raise ValueError(
            "NeuralTE feature-name count does not match attribution length: "
            f"{len(feature_names)} names versus {len(attr)} attribution values."
        )

    df = pd.DataFrame({
        "feature_index": np.arange(len(attr), dtype=int),
        "feature_name": feature_names,
        "feature_group": feature_group_names,
        "mean_abs_attribution": attr,
    })
    df["importance_rank"] = (
        df["mean_abs_attribution"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    df = df.sort_values("feature_index")
    df.to_csv(out_csv, index=False)


def run_neuralte(args, superf_dict, inv_superf_dict, num_classes):
    from Classifiers import NeuralTE

    print("[INFO] Running NeuralTE interpretability")

    model_path = os.path.join(args.models_dir, "NeuralTE_retrained_model.keras")
    model = load_model(
        model_path,
        custom_objects={"f1_m": NeuralTE.f1_m},
        compile=False,
    )

    NeuralTE.all_wicker_class = superf_dict
    NeuralTE.class_num = num_classes
    NeuralTE.inverted_all_wicker_class = {value: key for key, value in superf_dict.items()}

    internal_kmer_sizes = [1, 3]
    terminal_kmer_sizes = [1, 2, 3]
    project_dir = os.path.dirname(os.path.abspath(__file__))

    X, Y, seq_names, _, labels = NeuralTE.load_data(
        internal_kmer_sizes,
        terminal_kmer_sizes,
        args.fasta,
        args.work_dir,
        project_dir,
        args.threads,
    )

    X = X.astype("float32")

    preds = model.predict(X, batch_size=args.batch_size, verbose=0)
    pred_classes = np.argmax(preds, axis=1)

    selected_idx = np.arange(min(args.n_samples, X.shape[0]))
    X_sel = X[selected_idx]
    target_sel = pred_classes[selected_idx]

    attrs = integrated_gradients_keras(
        model,
        X_sel,
        target_class=tf.convert_to_tensor(target_sel),
        steps=args.steps,
    )[0]

    out_dir = os.path.join(args.out_dir, "NeuralTE")
    ensure_dir(out_dir)

    np.save(os.path.join(out_dir, "integrated_gradients.npy"), attrs)

    feature_groups = [
        ("Internal k-mers", 0, 68, "#4C78A8"),
        ("Terminal k-mers", 68, 236, "#F58518"),
        ("Sequence ends", 236, 276, "#54A24B"),
        ("Domain features", 276, 305, "#B279A2"),
    ]

    plot_neuralte_feature_attribution(
        attrs,
        feature_groups,
        os.path.join(out_dir, "NeuralTE_feature_attribution.png"),
        "NeuralTE Integrated Gradients",
    )

    save_feature_group_attributions(
        attrs,
        [(name, start, end) for name, start, end, color in feature_groups],
        os.path.join(out_dir, "NeuralTE_feature_group_attributions.csv"),
    )

    neuralte_feature_names, neuralte_feature_group_names = build_neuralte_feature_names(
        internal_kmer_sizes,
        terminal_kmer_sizes,
        NeuralTE.all_wicker_class_original,
    )
    save_neuralte_feature_attributions(
        attrs,
        neuralte_feature_names,
        neuralte_feature_group_names,
        os.path.join(out_dir, "NeuralTE_feature_attributions.csv"),
    )

    print("[INFO] NeuralTE interpretability completed")


def run_create(args, superf_dict, inv_superf_dict, num_classes):
    from Classifiers import CREATE

    print("[INFO] Running CREATE interpretability")

    model_path = os.path.join(args.models_dir, "CREATE_retrained_model.keras")
    CREATE.superf_dict = superf_dict

    k = 7
    l = 600

    X_kmer = CREATE.get_kmer_data(args.fasta, k)
    X_kmer = X_kmer.reshape(X_kmer.shape[0], 1, 4 ** k, 1).astype("float32")
    X_oh = CREATE.get_oh_data(args.fasta, l).astype("float32")
    labels = CREATE.get_label_data(args.fasta, mode="P").tolist()

    model, attention_model = CREATE.create_attn_model(k, l, num_classes)
    model.load_weights(model_path)

    preds = model.predict((X_kmer, X_oh), batch_size=args.batch_size, verbose=0)
    pred_classes = np.argmax(preds, axis=1)

    selected_idx = np.arange(min(args.n_samples, X_kmer.shape[0]))
    X_kmer_sel = X_kmer[selected_idx]
    X_oh_sel = X_oh[selected_idx]
    target_sel = pred_classes[selected_idx]

    attrs_kmer, attrs_oh = integrated_gradients_keras(
        model,
        [X_kmer_sel, X_oh_sel],
        target_class=tf.convert_to_tensor(target_sel),
        steps=args.steps,
    )

    out_dir = os.path.join(args.out_dir, "CREATE")
    ensure_dir(out_dir)

    np.save(os.path.join(out_dir, "CREATE_kmer_integrated_gradients.npy"), attrs_kmer)
    np.save(os.path.join(out_dir, "CREATE_onehot_integrated_gradients.npy"), attrs_oh)

    kmer_names = [
        "".join(k)
        for k in product("ACGT", repeat=7)
    ]

    plot_vector_attribution(
        attrs_kmer.reshape(attrs_kmer.shape[0], -1, 1),
        os.path.join(out_dir, "CREATE_kmer_attribution.png"),
        "CREATE k-mer branch Integrated Gradients",
        feature_names=kmer_names,
        top_k=5,
    )

    plot_sequence_attribution(
        attrs_oh,
        os.path.join(out_dir, "CREATE_onehot_sequence_attribution.png"),
        "CREATE one-hot branch Integrated Gradients",
    )

    branch_summary = pd.DataFrame([
        {
            "branch": "kmer_branch",
            "mean_abs_attribution": float(np.mean(np.abs(attrs_kmer))),
            "sum_abs_attribution": float(np.sum(np.abs(attrs_kmer))),
        },
        {
            "branch": "onehot_sequence_branch",
            "mean_abs_attribution": float(np.mean(np.abs(attrs_oh))),
            "sum_abs_attribution": float(np.sum(np.abs(attrs_oh))),
        },
    ])

    branch_summary.to_csv(
        os.path.join(out_dir, "CREATE_branch_attribution_summary.csv"),
        index=False,
    )

    print("[INFO] CREATE interpretability completed")


def run_terrier(args, superf_dict, inv_superf_dict, num_classes):
    print("[INFO] Running Terrier interpretability")

    try:
        import torch
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        raise ImportError(
            "Terrier interpretability requires Captum. Install it with: pip install captum"
        )

    import pickle
    from hierarchicalsoftmax import greedy_predictions
    from Classifiers import Terrier

    out_dir = os.path.join(args.out_dir, "Terrier")
    ensure_dir(out_dir)

    model_path = os.path.join(args.models_dir, "Terrier_retrained_model.pt")
    root_path = os.path.join(args.models_dir, "root.pkl")

    max_len = args.terrier_max_len
    batch_size = args.batch_size

    X, labels, _ = Terrier.load_data(args.fasta, max_len, mode="P")
    labels = labels.tolist()

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    with open(root_path, "rb") as f:
        root = pickle.load(f)

    vocab_size = len("ACGTN") + 1

    model = Terrier.TerrierNet(root=root, vocab_size=vocab_size).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        dummy = torch.zeros(1, X.shape[1], dtype=torch.long, device=device)
        _ = model(dummy)

    X_dataset = Terrier.InferenceDataset(X)
    X_loader = torch.utils.data.DataLoader(
        X_dataset,
        batch_size=batch_size,
        num_workers=0
    )

    test_logits_list = []
    all_probs = []

    with torch.no_grad():
        for xb in X_loader:
            xb = xb.to(device)
            logits = model(xb)
            test_logits_list.append(logits.cpu())

            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    test_logits = torch.cat(test_logits_list, dim=0)
    pred_nodes = greedy_predictions(test_logits, root)
    y_preds_probs = np.concatenate(all_probs, axis=0)

    selected_idx = np.arange(min(args.n_samples, X.shape[0]))
    X_sel_np = X[selected_idx]
    labels_sel = [labels[i] for i in selected_idx]
    pred_nodes_sel = [pred_nodes[i] for i in selected_idx]

    X_tensor = torch.tensor(X_sel_np, dtype=torch.long, device=device)

    def forward_func(input_ids):
        return model(input_ids)

    with torch.no_grad():
        logits_sel = forward_func(X_tensor)
        target = torch.argmax(logits_sel, dim=1)

    lig = LayerIntegratedGradients(forward_func, model.embedding)

    attributions = lig.attribute(
        X_tensor,
        target=target,
        n_steps=args.steps
    )

    attr = attributions.detach().cpu().numpy()

    attr_pos = np.abs(attr).sum(axis=-1)
    mean_attr_pos = attr_pos.mean(axis=0)

    np.save(os.path.join(out_dir, "Terrier_integrated_gradients.npy"), attr)
    np.save(os.path.join(out_dir, "Terrier_position_attribution.npy"), attr_pos)
    np.save(os.path.join(out_dir, "Terrier_prediction_probabilities.npy"), y_preds_probs[selected_idx])

    pd.DataFrame({
        "sequence_id": labels_sel,
        "predicted_node": [str(x) for x in pred_nodes_sel],
    }).to_csv(
        os.path.join(out_dir, "Terrier_selected_predictions.csv"),
        index=False
    )

    plt.figure(figsize=(12, 4))
    plt.plot(mean_attr_pos)
    plt.xlabel("Sequence position")
    plt.ylabel("Mean absolute attribution")
    plt.title("Terrier Integrated Gradients over embedding layer")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Terrier_position_attribution.png"), dpi=300)
    plt.close()

    top_positions = np.argsort(mean_attr_pos)[::-1][:args.top_k_positions]

    pd.DataFrame({
        "rank": np.arange(1, len(top_positions) + 1),
        "position": top_positions,
        "mean_abs_attribution": mean_attr_pos[top_positions],
    }).to_csv(
        os.path.join(out_dir, "Terrier_top_attributed_positions.csv"),
        index=False
    )

    print("[INFO] Terrier interpretability completed")


def run_deepte(args, superf_dict, inv_superf_dict, num_classes):
    original_set_memory_growth = tf.config.experimental.set_memory_growth

    def safe_set_memory_growth(device, enable):
        try:
            original_set_memory_growth(device, enable)
        except RuntimeError as e:
            print(f"[WARNING] TensorFlow memory growth already initialized: {e}")

    tf.config.experimental.set_memory_growth = safe_set_memory_growth

    try:
        from Classifiers import DeepTE
    finally:
        tf.config.experimental.set_memory_growth = original_set_memory_growth

    print("[INFO] Running DeepTE interpretability")

    out_dir = os.path.join(args.out_dir, "DeepTE")
    ensure_dir(out_dir)

    model_path = os.path.join(args.models_dir, "DeepTE_retrained_model.keras")
    if not os.path.exists(model_path):
        alt_model_path = os.path.join(args.models_dir, "DeepTE_retrained_model.weights.h5")
        if os.path.exists(alt_model_path):
            model_path = alt_model_path
        else:
            raise FileNotFoundError(
                f"Could not find DeepTE model weights/model at {model_path} or {alt_model_path}"
            )

    DeepTE.superf_dict = superf_dict

    X, labels = DeepTE.load_data(args.fasta, mode="P")
    labels = labels.tolist()

    X = X.astype("float32")
    X = X.reshape(X.shape[0], 1, 4 ** 7, 1)

    try:
        model = load_model(
            model_path,
            custom_objects={"f1_m": DeepTE.f1_m},
            compile=False,
        )
    except Exception as e:
        print(f"[WARNING] Could not load DeepTE as a full Keras model: {e}")
        print("[INFO] Rebuilding DeepTE architecture and loading weights")
        model = DeepTE.get_model(num_classes)
        model.load_weights(model_path)

    preds = model.predict(X, batch_size=args.batch_size, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    pred_probs = np.max(preds, axis=1)

    selected_idx = np.arange(min(args.n_samples, X.shape[0]))
    X_sel = X[selected_idx]
    target_sel = pred_classes[selected_idx]

    attrs = integrated_gradients_keras(
        model,
        X_sel,
        target_class=tf.convert_to_tensor(target_sel),
        steps=args.steps,
    )[0]

    np.save(os.path.join(out_dir, "DeepTE_integrated_gradients.npy"), attrs)
    np.save(os.path.join(out_dir, "DeepTE_prediction_probabilities.npy"), preds[selected_idx])

    kmer_names = [
        "".join(k)
        for k in product("ACGT", repeat=7)
    ]

    plot_vector_attribution(
        attrs,
        os.path.join(out_dir, "DeepTE_attribution.png"),
        "DeepTE Integrated Gradients",
        feature_names=kmer_names,
        top_k=5,
    )

    kmer_order = list(DeepTE.generate_kmer_dic(7).keys())
    mean_attr = np.abs(attrs).mean(axis=0).reshape(-1)
    top_idx = np.argsort(mean_attr)[::-1][:args.top_k_positions]

    pd.DataFrame({
        "rank": np.arange(1, len(top_idx) + 1),
        "feature_index": top_idx,
        "kmer": [kmer_order[i] for i in top_idx],
        "mean_abs_attribution": mean_attr[top_idx],
    }).to_csv(
        os.path.join(out_dir, "DeepTE_top_attributed_kmers.csv"),
        index=False,
    )

    pd.DataFrame({
        "sequence_id": [labels[i] for i in selected_idx],
        "predicted_class_index": pred_classes[selected_idx],
        "predicted_class": [inv_superf_dict.get(int(c), str(c)) for c in pred_classes[selected_idx]],
        "predicted_probability": pred_probs[selected_idx],
    }).to_csv(
        os.path.join(out_dir, "DeepTE_selected_predictions.csv"),
        index=False,
    )

    print("[INFO] DeepTE interpretability completed")


def load_config(json_path):
    if not os.path.exists(json_path):
        error(f"The configuration JSON file {json_path} was not found.")
    with open(json_path, 'r') as f:
        data = json.load(f)

    superf_dict = data.get('superf_dict', {})
    inv_superf_dict = {int(k): v for k, v in data.get('inv_superf_dict', {}).items()}
    num_classes = data.get('num_classes', 0)
    min_prob = data.get('min_prob', 0.0)
    species_group = data.get('species_group', 'unknown')

    return superf_dict, inv_superf_dict, num_classes, min_prob, species_group

def main():
    parser = argparse.ArgumentParser(
        description="Integrated Gradients interpretability analysis for PanTEon models"
    )

    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--models_dir", required=True, help="Directory containing trained PanTEon models")
    parser.add_argument("--work_dir", required=True, help="Working directory used by PanTEon feature extraction")
    parser.add_argument("--out_dir", default="interpretability_results", help="Output directory")
    parser.add_argument("--models", default="NeuralTE,CREATE,Terrier,DeepTE", help="Comma-separated model list")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--terrier_max_len", type=int, default=15000)
    parser.add_argument("--top_k_positions", type=int, default=50)

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    superf_dict, inv_superf_dict, num_classes, min_prob, species_group = load_config(
        f"{args.models_dir}/training_variables.json")

    selected_models = [m.strip() for m in args.models.split(",")]

    if "NeuralTE" in selected_models:
        run_neuralte(args, superf_dict, inv_superf_dict, num_classes)

    if "CREATE" in selected_models:
        run_create(args, superf_dict, inv_superf_dict, num_classes)

    if "Terrier" in selected_models:
        run_terrier(args, superf_dict, inv_superf_dict, num_classes)

    if "DeepTE" in selected_models:
        run_deepte(args, superf_dict, inv_superf_dict, num_classes)

    print("[INFO] All interpretability analyses completed")


if __name__ == "__main__":
    main()
