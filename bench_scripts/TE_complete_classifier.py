#!/usr/bin/env python3
"""
Classify consensus transposable elements as complete, incomplete, or uncertain
from their declared classification, sequence length, terminal structures, and
protein domains.

The domain HMM database should be the same curated profile database used by
MCHelper (for example Pfam35.0.hmm distributed with MCHelper), because domain
recognition relies on profile-name tags such as _RT_, _INT_, _Tase_, etc.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


DOMAIN_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "GAG": ("_GAG_",),
    "AP": ("_AP_", "ASPARTIC_PROTEASE"),
    "INT": ("_INT_",),
    "RT": ("_RT_", "_RVT_", "REVERSE_TRANSCRIPTASE"),
    "RNASEH": ("_RNASEH_", "RNASE_H"),
    "ENV": ("_ENV_",),
    "PHAGE_INT": ("_PHAGEINT_",),
    "EN": ("_EN_", "ENDONUCLEASE"),
    "TASE": ("_TASE_", "TRANSPOSASE"),
    "HEL": ("_HEL_", "HELICASE"),
    "RPA": ("_RPA_",),
    "REP": ("_REP_",),
    "OTU": ("_OTU_",),
    "SET": ("_SET_",),
    "PRP": ("_PRP",),
    "ATPASE": ("_ATPASE_",),
}

ORDER_TO_SUPERFAMILIES: Dict[str, Set[str]] = {
    "LTR": {"COPIA", "GYPSY", "BELPAO", "BEL-PAO", "ERV"},
    "LINE": {"R2", "RTE", "JOCKEY", "L1", "I", "R1", "CR1", "LOA", "L2"},
    "SINE": {"SINE"},
    "DIRS": {"DIRS", "NGARO", "VIPER"},
    "PLE": {"PLE", "PENELOPE"},
    "TIR": {
        "TC1MARINER", "TC1-MARINER", "HAT", "MUTATOR", "MERLIN", "TRANSIB",
        "P", "PIGGYBAC", "PIFHARBINGER", "PIF-HARBINGER", "CACTA", "MULE", "CMC",
    },
    "MITE": {"MITE"},
    "HELITRON": {"HELITRON"},
    "MAVERICK": {"MAVERICK", "POLINTON"},
    "CRYPTON": {"CRYPTON"},
    "TRIM": {"TRIM"},
    "LARD": {"LARD"},
}

CLASS_NAMES = {"CLASSI", "CLASS1", "CLASSII", "CLASS2", "RETROTRANSPOSON", "TRANSPOSON"}
ORDER_NAMES = set(ORDER_TO_SUPERFAMILIES)

# Broad biological ranges used only as supporting evidence, not hard universal definitions.
DEFAULT_LENGTH_RANGES: Dict[str, Tuple[int, int]] = {
    "LTR": (3_000, 20_000),
    "TRIM": (100, 2_500),
    "LARD": (2_000, 10_000),
    "LINE": (2_000, 12_000),
    "SINE": (80, 1_000),
    "DIRS": (3_000, 15_000),
    "PLE": (2_000, 8_000),
    "TIR": (500, 20_000),
    "MITE": (50, 1_500),
    "HELITRON": (1_000, 25_000),
    "MAVERICK": (10_000, 40_000),
    "CRYPTON": (1_000, 10_000),
}


@dataclass
class Classification:
    raw: str
    te_class: str
    order: str
    superfamily: str


@dataclass
class TerminalFeatures:
    ltr_length: int = 0
    tir_length: int = 0
    polya_length: int = 0
    self_hits: int = 0


@dataclass
class Assessment:
    status: str
    score: int
    reason: str
    required_present: List[str]
    required_missing: List[str]


def run_command(command: Sequence[str], *, stdout=None, stderr=None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, check=True, stdout=stdout, stderr=stderr, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        cmd = " ".join(command)
        raise RuntimeError(f"Command failed ({exc.returncode}): {cmd}") from exc


def check_dependencies() -> None:
    missing = [exe for exe in ("makeblastdb", "blastn", "getorf", "hmmpress", "hmmscan") if shutil.which(exe) is None]
    if missing:
        raise RuntimeError("Missing external dependencies: " + ", ".join(missing))


def normalize_token(value: str) -> str:
    return value.strip().upper().replace(" ", "").replace("_", "-")


def infer_order(token: str) -> Optional[str]:
    t = normalize_token(token)
    if t in ORDER_NAMES:
        return t
    for order, families in ORDER_TO_SUPERFAMILIES.items():
        if t in {normalize_token(x) for x in families}:
            return order
    if t == "DNA":
        return "TIR"
    return None


def parse_classification(record: SeqRecord) -> Classification:
    identifier = record.id
    if "#" not in identifier:
        return Classification("UNCLASSIFIED", "UNCLASSIFIED", "UNCLASSIFIED", "UNCLASSIFIED")

    raw = identifier.split("#", 1)[1]
    parts = [normalize_token(x) for x in raw.split("/") if x.strip()]
    te_class = "UNCLASSIFIED"
    order = "UNCLASSIFIED"
    superfamily = "UNCLASSIFIED"

    if len(parts) >= 3:
        te_class, order, superfamily = parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        first_order = infer_order(parts[0])
        second_order = infer_order(parts[1])
        if parts[0] in CLASS_NAMES:
            te_class, order = parts[0], second_order or parts[1]
        elif first_order:
            order, superfamily = first_order, parts[1]
        elif second_order:
            order, superfamily = second_order, parts[1]
    elif len(parts) == 1:
        order = infer_order(parts[0]) or "UNCLASSIFIED"
        if order != parts[0]:
            superfamily = parts[0]

    inferred = infer_order(superfamily)
    if inferred:
        order = inferred
    else:
        order = infer_order(order) or order

    if te_class == "UNCLASSIFIED":
        if order in {"LTR", "TRIM", "LARD", "LINE", "SINE", "DIRS", "PLE"}:
            te_class = "CLASSI"
        elif order in {"TIR", "MITE", "HELITRON", "MAVERICK", "CRYPTON"}:
            te_class = "CLASSII"

    return Classification(raw, te_class, order, superfamily)


def poly_tail_length(sequence: str, minimum: int) -> int:
    seq = sequence.upper()
    allowance = max(1, minimum // 8)
    best = 0
    for segment in (seq[: minimum + allowance], seq[-(minimum + allowance):]):
        for base in ("A", "T"):
            count = 0
            mismatches = 0
            iterator = segment if segment is seq[: minimum + allowance] else segment[::-1]
            for char in iterator:
                if char == base:
                    count += 1
                else:
                    mismatches += 1
                    if mismatches > allowance:
                        break
            if count >= minimum:
                best = max(best, count)
    return best


def detect_terminal_repeats(record: SeqRecord, temp_dir: Path, min_ltr: int, min_tir: int,
                            min_polya: int, threads: int, terminal_fraction: float) -> TerminalFeatures:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.id.split("#", 1)[0])
    fasta_path = temp_dir / f"{safe_name}.fa"
    blast_path = temp_dir / f"{safe_name}.self.tsv"
    SeqIO.write([record], fasta_path, "fasta")

    run_command(["makeblastdb", "-in", str(fasta_path), "-dbtype", "nucl"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_command([
        "blastn", "-query", str(fasta_path), "-db", str(fasta_path),
        "-num_threads", str(threads), "-out", str(blast_path),
        "-outfmt", "6 qstart qend sstart send length pident",
        "-word_size", "11", "-gapopen", "5", "-gapextend", "2",
        "-reward", "2", "-penalty", "-3",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    length = len(record.seq)
    edge = max(1, int(length * terminal_fraction))
    best_ltr = 0
    best_tir = 0
    hit_count = 0

    with blast_path.open() as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) < 6:
                continue
            qstart, qend, sstart, send, aln_len = map(int, fields[:5])
            pident = float(fields[5])
            # Ignore the full-length identity diagonal.
            if min(qstart, qend, sstart, send) == 1 and max(qstart, qend, sstart, send) == length:
                continue
            hit_count += 1
            coords = (qstart, qend, sstart, send)
            spans_both_ends = min(coords) <= edge and max(coords) >= length - edge
            if not spans_both_ends or pident < 70.0:
                continue
            same_orientation = (qend - qstart) * (send - sstart) > 0
            if same_orientation and aln_len >= min_ltr:
                best_ltr = max(best_ltr, aln_len)
            elif not same_orientation and aln_len >= min_tir:
                best_tir = max(best_tir, aln_len)

    return TerminalFeatures(
        ltr_length=best_ltr,
        tir_length=best_tir,
        polya_length=poly_tail_length(str(record.seq), min_polya),
        self_hits=hit_count,
    )


def ensure_hmm_pressed(hmm_db: Path) -> None:
    required = [Path(str(hmm_db) + suffix) for suffix in (".h3f", ".h3i", ".h3m", ".h3p")]
    if not all(path.exists() for path in required):
        run_command(["hmmpress", str(hmm_db)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def canonical_domain(profile_name: str) -> Optional[str]:
    upper = profile_name.upper().replace("-", "_")
    for domain, patterns in DOMAIN_PATTERNS.items():
        if any(pattern in upper for pattern in patterns):
            return domain
    return None


def detect_domains(record: SeqRecord, hmm_db: Path, temp_dir: Path, evalue: float,
                   threads: int, min_orf_nt: int) -> Tuple[Set[str], List[str]]:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.id.split("#", 1)[0])
    fasta_path = temp_dir / f"{safe_name}.fa"
    orf_path = temp_dir / f"{safe_name}.orfs.fa"
    table_path = temp_dir / f"{safe_name}.hmmscan.tbl"
    SeqIO.write([record], fasta_path, "fasta")

    run_command(["getorf", "-sequence", str(fasta_path), "-minsize", str(min_orf_nt), "-outseq", str(orf_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not orf_path.exists() or orf_path.stat().st_size == 0:
        return set(), []

    run_command([
        "hmmscan", "--tblout", str(table_path), "-E", str(evalue), "--noali",
        "--cpu", str(threads), str(hmm_db), str(orf_path),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    domains: Set[str] = set()
    raw_profiles: List[str] = []
    if table_path.exists():
        with table_path.open() as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.split()
                if len(fields) < 5:
                    continue
                profile = fields[0]
                try:
                    full_evalue = float(fields[4])
                except ValueError:
                    continue
                if full_evalue <= evalue:
                    raw_profiles.append(profile)
                    domain = canonical_domain(profile)
                    if domain:
                        domains.add(domain)
    return domains, sorted(set(raw_profiles))


def length_support(order: str, length: int) -> Tuple[bool, str]:
    if order not in DEFAULT_LENGTH_RANGES:
        return True, "no class-specific length range"
    low, high = DEFAULT_LENGTH_RANGES[order]
    return low <= length <= high, f"expected broad range {low}-{high} bp"


def assess_element(classification: Classification, length: int, terminal: TerminalFeatures,
                   domains: Set[str], min_ltr_domains: int) -> Assessment:
    order = classification.order.upper()
    present: List[str] = []
    missing: List[str] = []
    score = 0
    len_ok, len_note = length_support(order, length)
    score += 1 if len_ok else 0

    def has(label: str, condition: bool, weight: int = 1) -> None:
        nonlocal score
        (present if condition else missing).append(label)
        if condition:
            score += weight

    # Rules are intentionally conservative: COMPLETE requires the principal
    # hallmarks expected for an autonomous full-length element. Non-autonomous
    # orders (SINE/MITE/TRIM/LARD) use terminal hallmarks and length instead.
    if order == "LTR":
        ltr_core = {"RT", "INT", "RNASEH"}
        has("paired LTRs", terminal.ltr_length > 0, 2)
        has("RT domain", "RT" in domains, 2)
        has("integrase domain", "INT" in domains)
        has("RNase H domain", "RNASEH" in domains)
        good_domains = len(domains & {"GAG", "AP", "INT", "RT", "RNASEH", "ENV"})
        has(f">={min_ltr_domains} LTR-associated domains", good_domains >= min_ltr_domains)
        complete = terminal.ltr_length > 0 and "RT" in domains and good_domains >= min_ltr_domains
    elif order in {"TRIM", "LARD"}:
        has("paired LTRs", terminal.ltr_length > 0, 2)
        has("absence of coding domains", len(domains) == 0)
        complete = terminal.ltr_length > 0 and len(domains) == 0 and len_ok
    elif order == "LINE":
        has("RT domain", "RT" in domains, 2)
        has("endonuclease/RNase H domain", bool(domains & {"EN", "RNASEH"}))
        has("3-prime poly(A/T) tail", terminal.polya_length > 0)
        complete = "RT" in domains and bool(domains & {"EN", "RNASEH"}) and len_ok
    elif order == "SINE":
        has("3-prime poly(A/T) tail", terminal.polya_length > 0, 2)
        has("absence of autonomous protein domains", not bool(domains & {"RT", "EN", "TASE"}))
        complete = terminal.polya_length > 0 and not bool(domains & {"RT", "EN", "TASE"}) and len_ok
    elif order == "DIRS":
        has("RT domain", "RT" in domains, 2)
        has("tyrosine-recombinase/phage-integrase domain", "PHAGE_INT" in domains, 2)
        has("RNase H or GAG domain", bool(domains & {"RNASEH", "GAG"}))
        complete = "RT" in domains and "PHAGE_INT" in domains and len_ok
    elif order == "PLE":
        has("RT domain", "RT" in domains, 2)
        has("endonuclease domain", "EN" in domains, 2)
        complete = "RT" in domains and "EN" in domains and len_ok
    elif order == "TIR":
        has("paired TIRs", terminal.tir_length > 0, 2)
        has("transposase domain", "TASE" in domains, 2)
        complete = terminal.tir_length > 0 and "TASE" in domains and len_ok
    elif order == "MITE":
        has("paired TIRs", terminal.tir_length > 0, 2)
        has("absence of transposase domain", "TASE" not in domains)
        complete = terminal.tir_length > 0 and "TASE" not in domains and len_ok
    elif order == "HELITRON":
        has("Rep domain", "REP" in domains, 2)
        has("helicase domain", "HEL" in domains, 2)
        complete = "REP" in domains and "HEL" in domains and len_ok
    elif order == "MAVERICK":
        has("protein-primed polymerase domain", "PRP" in domains, 2)
        has("ATPase domain", "ATPASE" in domains)
        has("integrase domain", "INT" in domains)
        complete = "PRP" in domains and bool(domains & {"ATPASE", "INT"}) and len_ok
    elif order == "CRYPTON":
        has("tyrosine-recombinase/phage-integrase domain", "PHAGE_INT" in domains, 2)
        complete = "PHAGE_INT" in domains and len_ok
    else:
        return Assessment(
            "UNCERTAIN", score,
            "Classification is missing or unsupported; completeness cannot be evaluated reliably.",
            present, missing,
        )

    if complete:
        status = "COMPLETE"
        reason = f"Principal structural/coding hallmarks for {order} are present; length: {len_note}."
    elif present and missing:
        status = "INCOMPLETE"
        reason = f"One or more principal hallmarks expected for {order} are missing; length: {len_note}."
    else:
        status = "UNCERTAIN"
        reason = f"Insufficient diagnostic evidence for {order}; length: {len_note}."
    if not len_ok and status == "COMPLETE":
        status = "UNCERTAIN"
        reason = f"Structural hallmarks are present, but length is outside the {order} broad reference range ({len_note})."
    return Assessment(status, score, reason, present, missing)


def write_fasta(records: List[SeqRecord], path: Path) -> None:
    SeqIO.write(records, path, "fasta")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect TE structural hallmarks and classify consensus sequences as complete/incomplete.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input classified TE FASTA")
    parser.add_argument("-p", "--profiles", required=True, type=Path, help="HMM profile database used by MCHelper")
    parser.add_argument("-o", "--output", default=Path("te_completeness"), type=Path, help="Output directory")
    parser.add_argument("--threads", type=int, default=1, help="Threads for BLAST/HMMER per sequence")
    parser.add_argument("--min-ltr", type=int, default=20, help="Minimum detected LTR length")
    parser.add_argument("--min-tir", type=int, default=10, help="Minimum detected TIR length")
    parser.add_argument("--min-polya", type=int, default=10, help="Minimum poly(A/T) tail length")
    parser.add_argument("--min-ltr-domains", type=int, default=3, help="Minimum LTR-associated domain types")
    parser.add_argument("--min-orf-nt", type=int, default=300, help="Minimum ORF length passed to getorf")
    parser.add_argument("--hmm-evalue", type=float, default=1e-5, help="Maximum HMM full-sequence E-value")
    parser.add_argument("--terminal-fraction", type=float, default=0.10,
                        help="Fraction at each sequence edge considered terminal")
    parser.add_argument("--keep-temp", action="store_true", help="Retain per-sequence temporary files")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input FASTA not found: {args.input}")
    if not args.profiles.is_file():
        parser.error(f"HMM database not found: {args.profiles}")
    if args.threads < 1:
        parser.error("--threads must be >= 1")

    try:
        check_dependencies()
        args.output.mkdir(parents=True, exist_ok=True)
        ensure_hmm_pressed(args.profiles)

        records = list(SeqIO.parse(args.input, "fasta"))
        if not records:
            raise RuntimeError("The input FASTA contains no sequences.")

        complete_records: List[SeqRecord] = []
        incomplete_records: List[SeqRecord] = []
        uncertain_records: List[SeqRecord] = []
        rows: List[Dict[str, object]] = []

        temp_context = tempfile.TemporaryDirectory(prefix="te_complete_", dir=args.output)
        temp_dir = Path(temp_context.name)

        for index, record in enumerate(records, start=1):
            seq_name = record.id.split("#", 1)[0]
            print(f"[{index}/{len(records)}] {seq_name}", file=sys.stderr)
            classification = parse_classification(record)
            terminal = detect_terminal_repeats(
                record, temp_dir, args.min_ltr, args.min_tir, args.min_polya,
                args.threads, args.terminal_fraction,
            )
            domains, raw_profiles = detect_domains(
                record, args.profiles, temp_dir, args.hmm_evalue,
                args.threads, args.min_orf_nt,
            )
            assessment = assess_element(
                classification, len(record.seq), terminal, domains, args.min_ltr_domains,
            )

            annotated = record[:]
            annotated.id = (
                f"{record.id}|completeness={assessment.status}|"
                f"LTR={terminal.ltr_length}|TIR={terminal.tir_length}|polyA={terminal.polya_length}|"
                f"domains={','.join(sorted(domains)) or 'NA'}"
            )
            annotated.description = ""
            if assessment.status == "COMPLETE":
                complete_records.append(annotated)
            elif assessment.status == "INCOMPLETE":
                incomplete_records.append(annotated)
            else:
                uncertain_records.append(annotated)

            rows.append({
                "sequence_id": seq_name,
                "original_header": record.id,
                "length_bp": len(record.seq),
                "class": classification.te_class,
                "order": classification.order,
                "superfamily": classification.superfamily,
                "ltr_length": terminal.ltr_length,
                "tir_length": terminal.tir_length,
                "polyA_T_length": terminal.polya_length,
                "self_blast_hits": terminal.self_hits,
                "domains": ",".join(sorted(domains)) or "NA",
                "raw_hmm_profiles": ",".join(raw_profiles) or "NA",
                "status": assessment.status,
                "evidence_score": assessment.score,
                "required_present": "; ".join(assessment.required_present) or "NA",
                "required_missing": "; ".join(assessment.required_missing) or "NA",
                "reason": assessment.reason,
            })

        report_path = args.output / "te_completeness.tsv"
        with report_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

        write_fasta(complete_records, args.output / "complete_elements.fa")
        write_fasta(incomplete_records, args.output / "incomplete_elements.fa")
        write_fasta(uncertain_records, args.output / "uncertain_elements.fa")

        summary_path = args.output / "summary.txt"
        with summary_path.open("w") as handle:
            handle.write(f"Total\t{len(records)}\n")
            handle.write(f"Complete\t{len(complete_records)}\n")
            handle.write(f"Incomplete\t{len(incomplete_records)}\n")
            handle.write(f"Uncertain\t{len(uncertain_records)}\n")

        if args.keep_temp:
            retained = args.output / "temp"
            if retained.exists():
                shutil.rmtree(retained)
            shutil.copytree(temp_dir, retained)
        temp_context.cleanup()

        print(f"Report: {report_path}", file=sys.stderr)
        return 0
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
