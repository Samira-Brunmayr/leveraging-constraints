"""Script to calculate the signal statistics of wick test data fast5 files (i.e. signal values per base)"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from Bio import SeqIO
from ont_fast5_api.fast5_interface import get_fast5_file

from main.scripts.constants import TEST_DATA, TRAIN_DATA


def get_reads(fast5file):
    fast5_filepath = fast5file
    with get_fast5_file(fast5_filepath, mode="r") as f5:
        for read in f5.get_reads():
            raw_data = read.get_raw_data()
            return read.read_id, raw_data


def process(fast5dir: str, basecalls: str) -> list[float]:
    id_to_signal = {}

    signals_per_base = []

    for f in os.listdir(fast5dir):
        if os.path.splitext(f)[1] == ".fast5":
            read_id, signal = get_reads(os.path.join(fast5dir, f))
            id_to_signal[read_id] = signal

    if os.path.splitext(basecalls)[1] == ".fastq":
        file_type = "fastq"
    elif os.path.splitext(basecalls)[1] == ".fasta":
        file_type = "fasta"

    for record in SeqIO.parse(basecalls, file_type):
        seq = str(record.seq)
        read_id = record.id
        if read_id in id_to_signal:
            signals_per_base.append(len(id_to_signal[read_id]) / len(seq))
    return signals_per_base


def main(wick_dir: str, output: str, threads: int) -> dict[str, list[float]]:
    if not os.path.isdir(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))

    folder_to_fast5 = {}
    folder_to_basecalls = {}
    folders: list[str] = []
    results = {}

    for folder in os.listdir(wick_dir):
        if folder in TEST_DATA or folder in TRAIN_DATA:
            folders.append(folder)
            fast5dir = os.path.join(wick_dir, folder, "fast5")

            if folder in TEST_DATA:
                basecalls = os.path.join(wick_dir, folder, "fastq", "basecalls.fasta")
            elif folder in TRAIN_DATA:
                basecalls = os.path.join(wick_dir, folder, "read_references.fastq")

            folder_to_fast5[folder] = fast5dir
            folder_to_basecalls[folder] = basecalls

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {
            executor.submit(process, folder_to_fast5[f], folder_to_basecalls[f]): f
            for f in folders
        }

        for future in as_completed(futures):
            f = futures[future]
            result = future.result()
            results[f] = result

    with open(output, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wick-dir", type=str, required=True, help="Location of wick data root dir"
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(wick_dir=args.wick_dir, output=args.output, threads=args.threads)
