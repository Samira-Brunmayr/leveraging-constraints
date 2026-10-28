"""Evaluation metrics calculations"""

import os
import random
from collections import Counter
from typing import Any, Generator

import mappy
import pandas as pd
from Bio import SeqIO

from main.file_encoder import Fragment
from main.nucleotide_string import NucleotideString


def to_fasta(fastq_path: str, fasta_path: str) -> None:
    """Convert a fastq file to fasta"""
    if not os.path.isfile(fastq_path):
        raise FileNotFoundError(f"Could not find file {fastq_path}")

    # make output directory if it doesnt exist
    output_dir = os.path.dirname(fasta_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(fasta_path, "w") as fasta_out:
        SeqIO.convert(
            in_file=fastq_path,
            in_format="fastq",
            out_file=fasta_out,
            out_format="fasta",
        )


def get_hits(
    aligner: mappy.Aligner,
    fastq_records: Any | Generator[Any, None, None],
    primary: bool = False,
) -> list[mappy.Alignment]:
    """Return the hits -> primary hits are usually best aligment match,
    for consensus we want all hits"""
    hits = []
    failed_aligns = 0
    for record in fastq_records:
        has_hit = False
        nucleotide_sequence = str(record.seq)
        for hit in aligner.map(nucleotide_sequence):
            has_hit = True
            if primary:
                if hit.is_primary:
                    hits.append((record.id, hit))
                    break
            else:
                hits.append((record.id, hit))
        if not has_hit:
            failed_aligns += 1
    return hits, failed_aligns


def get_cigar_counts(hit: mappy.Alignment) -> tuple[int, int, int, int]:
    """Get the count of matches, mismatches, insertions, deletions"""
    matches_and_mismatches = 0
    insertions = 0
    deletions = 0
    for count, op in hit.cigar:
        if op == 0:
            # M = match/mismatch
            matches_and_mismatches += count
        if op == 1:
            # I = insertion
            insertions += count
        if op == 2:
            # D = deletion
            deletions += count

    mismatches = hit.NM - insertions - deletions
    matches = matches_and_mismatches - mismatches

    return matches, mismatches, insertions, deletions


def get_metrics(hits: list[tuple[str, mappy.Alignment]]) -> pd.DataFrame:
    """return a dataframe with all the metrics we care about"""
    # columns:
    metrics = []
    for query_id, hit in hits:
        matches, mismatches, insertions, deletions = get_cigar_counts(hit)

        # accuracy is wrt the reference
        accuracy = matches / hit.ctg_len

        # rates are wrt alignment length
        alignment_length = matches + mismatches + insertions + deletions
        match_rate = matches / alignment_length
        mismatch_rate = mismatches / alignment_length
        insertion_rate = insertions / alignment_length
        deletion_rate = deletions / alignment_length

        metrics.append(
            {
                "Reference ID": hit.ctg,
                "Reference Length": hit.ctg_len,
                "Reference Start": hit.r_st,
                "Reference End": hit.r_en,
                "Query ID": query_id,
                "Query Start": hit.q_st,
                "Query End": hit.q_en,
                "MAPQ": hit.mapq,
                "blen": hit.blen,
                "mlen": hit.mlen,
                "Matches": matches,
                "Mismatches": mismatches,
                "Insertions": insertions,
                "Deletions": deletions,
                "Alignment Length": alignment_length,
                "Accuracy": accuracy,
                "Match Rate": match_rate,
                "Mismatch Rate": mismatch_rate,
                "Insertion Rate": insertion_rate,
                "Deletion Rate": deletion_rate,
            }
        )
    return pd.DataFrame(metrics)


def get_consensus_accuracy(
    reference_genome: str, consensus_assembly: str, chunk_size: int
) -> tuple[float, pd.DataFrame]:
    """Compute consensus accuracy as defined in https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-02903-2
    using the BLAST identity (M/M+X+I+D) - https://lh3.github.io/2018/11/25/on-the-definition-of-sequence-identity
    i.e. median of BLAST identities of each chunk"""
    aln = mappy.Aligner(reference_genome, preset="asm5")
    records = SeqIO.parse(consensus_assembly, "fasta")

    accuracies = []

    for record in records:
        seq = str(record.seq)
        chunks = [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]
        for idx, chunk in enumerate(chunks):
            hits = aln.map(chunk)
            for hit in hits:
                if hit.is_primary:
                    matches, mismatches, insertions, deletions = get_cigar_counts(hit)
                    accuracies.append(
                        {
                            "Chunk Number": idx,
                            "Chunk Accuracy": matches
                            / (matches + mismatches + insertions + deletions),
                        }
                    )

    df = pd.DataFrame(accuracies)
    return (df["Chunk Accuracy"].median(), df)


def merge(candidates: list[NucleotideString]) -> str:
    """Given multiple same length candidates, return a consensus"""
    if len(candidates) == 0:
        # nothing, just return arbitrary 25chars
        return "XXXXXXXXXXXXXXXXXXXXXXXXX"
    else:
        strings = [x.string for x in candidates]
        length = len(strings[0])
        result = []

        for i in range(length):
            chars_at_pos = [s[i] for s in strings]
            counts = Counter(chars_at_pos)
            max_count = max(counts.values())

            majority_chars = [
                char for char, count in counts.items() if count == max_count
            ]

            chosen_char = random.choice(majority_chars)
            result.append(chosen_char)
        return "".join(result)


def nanopore_decode_fragments(
    prediction_path: str,
    reference_path: str,
) -> str:
    """Given a fastq file with basecalled reads and the original fasta,
    attempt to reconstruct the payload DNA"""

    # build alignment
    aln = mappy.Aligner(reference_path, preset="map-ont")
    prediction_records = SeqIO.parse(prediction_path, "fastq")
    primary_hits, failed_aligns = get_hits(aln, prediction_records, primary=True)
    metrics = get_metrics(primary_hits)
    metrics = metrics[metrics["Query ID"] == metrics["Reference ID"]]
    hit_dict = {}
    for i, row in metrics.iterrows():
        hit_dict[row["Query ID"]] = (
            row["Query Start"],
            row["Query End"],
            row["Reference Start"],
            row["Reference End"],
        )

    # build dict of index to id
    index_to_id = {}
    for rec in SeqIO.parse(reference_path, "fasta"):
        index_to_id[int(rec.description.split()[1].split("_")[-1])] = rec.id
    index_to_id = dict(sorted(index_to_id.items()))

    # build dict of id to sequence
    id_to_basecall = {}
    for rec in SeqIO.parse(prediction_path, "fastq"):
        id_to_basecall[rec.id] = str(rec.seq)

    output = ""
    window: list[list[NucleotideString]] = [[], [], [], []]
    # go through index by index
    for dna_index, seq_id in index_to_id.items():
        if seq_id in hit_dict:
            sequence = NucleotideString(string=id_to_basecall[seq_id])
            query_start, query_end, reference_start, reference_end = hit_dict[seq_id]
            offset = reference_start - query_start

            chunk_indices = [(1, 26), (26, 51), (51, 76), (76, 101)]
            if dna_index % 2 == 1:
                chunk_indices = chunk_indices[::-1]
            chunks: list[NucleotideString | None] = [None, None, None, None]
            # get chunks
            for chunk_idx, (start, end) in enumerate(chunk_indices):
                if (
                    reference_start <= start
                    and reference_end >= end
                    and len(sequence) > end - offset
                ):
                    chunks[chunk_idx] = sequence[start - offset : end - offset]
            # revserse complement chunks if necessary
            if dna_index % 2 == 1:
                for i in range(0, len(chunks)):
                    if chunks[i] is not None:
                        chunks[i] = chunks[i].reverse().complement()
            # add chunks to window
            for i in range(0, len(window)):
                if chunks[i] is not None:
                    window[i].append(chunks[i])
        # shift window
        output += merge(window[0])
        window = window[1:]
        window.append([])
    output += merge(window[0])
    output += merge(window[1])
    output += merge(window[2])
    return output


def get_payload_accuracy(
    basecalls_file: str, fasta_file: str
) -> tuple[float, str, str]:
    """Compute the payload accuracy by reconstructing the encoded payload
    and comparing it to reference"""
    decoded_payload = nanopore_decode_fragments(basecalls_file, fasta_file)

    fragments = []
    for rec in SeqIO.parse(fasta_file, "fasta"):
        fragments.append(NucleotideString(string=str(rec.seq)))
    # construct ground truth payload
    fragments = list(map(lambda x: Fragment.from_nucleotide_string(x), fragments))
    ground_truth_payload = ""
    for i, fragment in enumerate(fragments):
        if i == 0:
            ground_truth_payload = fragment.data.string
        else:
            ground_truth_payload += fragment.data.string[75:]
    # compute accuracy
    correct = 0
    for i in range(0, len(decoded_payload)):
        if decoded_payload[i] == ground_truth_payload[i]:
            correct += 1
    return (correct / len(decoded_payload)), ground_truth_payload, decoded_payload
