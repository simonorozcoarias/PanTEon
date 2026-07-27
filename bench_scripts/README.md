# Benchmarking scripts used in the PanTEon's paper

## TE_complete_classifier.py

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
- `-i INPUT` : Path to the TE FASTA file (required).
- `-p PROFILES` : Path to the HMM profiles for detecting the TE coding domains. Use the file "Pfam35.0.hmm" released by the REPET group (https://urgi.versailles.inrae.fr/download/repet/profiles/ProfilesBankForREPET_Pfam35.0_GypsyDB.hmm.tar.gz) and used in the MCHelper pipeline (required).
- `-t THREADS, --threads THREADS` : Number of CPU threads to use
- `--min-ltr MIN_LTR`:
- `--min-tir MIN_TIR`:
- `--min-polya MIN_POLYA`:
- `--min-ltr-domains MIN_LTR_DOMAINS`:
- `--min-orf-nt MIN_ORF_NT`:
- `--hmm-evalue HMM_EVALUE`:
- `--terminal-fraction TERMINAL_FRACTION`:
- `--keep-temp`:
