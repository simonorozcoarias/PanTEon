# Benchmarking scripts used in the PanTEon's paper
This directory contains additional scripts developed and used during the development of the PanTEon framework and the experiments reported in the PanTEon paper. These scripts were primarily created to support the evaluation and demonstrate the capabilities of PanTEon, and therefore have not been as extensively tested as the main framework. Please use them with caution. If you encounter any bugs or unexpected behavior, we would greatly appreciate it if you could report them in the Issues section of this repository.

## TE_complete_classifier.py
Identify TE structural hallmarks and classify consensus sequences as structurally complete or incomplete according to the criteria established by MCHelper (Orozco-Arias et al., 2024), distinguishing intact elements from incomplete or degraded sequences.

```
usage:  TE_complete_classifier.py [-h] -i INPUT -p PROFILES [-o OUTPUT]
                                 [--threads THREADS] [--min-ltr MIN_LTR]
                                 [--min-tir MIN_TIR] [--min-polya MIN_POLYA]
                                 [--min-ltr-domains MIN_LTR_DOMAINS]
                                 [--min-orf-nt MIN_ORF_NT]
                                 [--hmm-evalue HMM_EVALUE]
                                 [--terminal-fraction TERMINAL_FRACTION]
                                 [--keep-temp]
```

### Options

- `-h, --help` : Show this help message and exit.
- `-i INPUT, --input INPUT` : Input classified TE FASTA (required).
- `-p PROFILES, --profiles PROFILES` : Path to the HMM profiles for detecting the TE coding domains. Use the file "Pfam35.0.hmm" released by the REPET group (https://urgi.versailles.inrae.fr/download/repet/profiles/ProfilesBankForREPET_Pfam35.0_GypsyDB.hmm.tar.gz) and used in the MCHelper pipeline (required).
- `-o OUTPUT, --output OUTPUT`: Output directory (default: te_completeness).
- `-t THREADS, --threads THREADS` : Threads for BLAST/HMMER per sequence (default: 1).
- `--min-ltr MIN_LTR`: Minimum detected LTR length (default: 20).
- `--min-tir MIN_TIR`: Minimum detected TIR length (default: 10).
- `--min-polya MIN_POLYA`: Minimum poly(A/T) tail length (default: 10).
- `--min-ltr-domains MIN_LTR_DOMAINS`:Minimum LTR-associated domain types for defying a LTR retrotransposon as "complete" (default: 3).
- `--min-orf-nt MIN_ORF_NT`: Minimum ORF length passed to getorf (default: 300).
- `--hmm-evalue HMM_EVALUE`: Maximum HMM full-sequence E-value (default: 1e-05).
- `--terminal-fraction TERMINAL_FRACTION`: Fraction at each sequence edge considered terminal (default: 0.1).
- `--keep-temp`: Retain per-sequence temporary files (default: False).

## interpret_panteon_models.py
Integrated Gradients interpretability analysis for PanTEon models

```
usage: interpret_panteon_models.py [-h] --fasta FASTA --models_dir MODELS_DIR
                                   --work_dir WORK_DIR [--out_dir OUT_DIR]
                                   [--models MODELS] [--threads THREADS]
                                   [--batch_size BATCH_SIZE]
                                   [--n_samples N_SAMPLES] [--steps STEPS]
                                   [--use_gpu] [--terrier_max_len TERRIER_MAX_LEN]
                                   [--top_k_positions TOP_K_POSITIONS]

```

### Options
- `-h, --help`: show this help message and exit.
- `--fasta FASTA`: Input FASTA file --models_dir MODELS_DIR Directory containing trained PanTEon models (required).
- `--work_dir WORK_DIR`: Working directory used by PanTEon feature extraction (required).
- `--out_dir OUT_DIR`: Output directory. Default="interpretability_results"
- `--models MODELS`: Comma-separated model list (required). Default="NeuralTE,CREATE,Terrier,DeepTE".
- `--threads THREADS`: Number of threads to be used by the script. Default=4.
- `--batch_size BATCH_SIZE`: batch size for the model inferences. Default=32.
- `--n_samples N_SAMPLES`: Number of samples to be used in the interpretability analysis. Default=50.
- `--steps STEPS`: Number of steps to be used in the integrated gradients for the keras-based models. Default=32
- `--use_gpu`: To use GPU or not for doing the model inference
- `--terrier_max_len TERRIER_MAX_LEN`: Maximum length used for trimming the sequences for being used by Terrier (should be the same as the one used for training). Default=15000.
- `--top_k_positions TOP_K_POSITIONS`: Number of top k-mer features to be save in the output files. Default=50.

## compute_f1_retraining.py
Generate an F1 heatmap + support bar plot and Critical Difference (CD) diagram from a wide-format Excel file with all the model's results.

```
usage: compute_f1_retraining.py model_results.xlsx [-o OUTPUT_CSV_FILE]
                                                   [--outdir OUTPUT_DIR] [--alpha]
                                                   [--formats FORMATS] [--sheet SHEET]

```

### Options
- `input INPUT`: Input Excel file (.xlsx/.xls). Required.
- `-o OUTPUT, --output OUTPUT`: Output CSV file (long format) with Tool, Superfamily, F1, and Support. Default="f1_superfamily_by_tool.csv".
- `--outdir OUTDIR`: Output directory for the figures. Default=figures.
- `--alpha ALPHA`: Significance level for CD (Nemenyi test). Default=0.05.
- `--formats FORMATS`: Figure output formats (choices are "pdf", "png", and "svg"). Default="pdf".
- `--sheet SHEET`: Excel sheet (name or index). Default=0.

## ensamble_classification.py
Script used for testing the esamble classification using three approaches: Majority voting system, Weighted voting system or Stacking method, based on XGBoosting.

## References
- Orozco-Arias, S., Sierra, P., Durbin, R., & González, J. (2024). MCHelper automatically curates transposable element libraries across eukaryotic species. Genome Research, 34(12), 2256-2268.
