"""Computes the payload accuracy for a dir containing basecalls.fastq"""

import os
from argparse import ArgumentParser

from main.models.metrics import get_payload_accuracy


def main(input_dir: str, goldman_dir: str):
    files = os.listdir(input_dir)
    for f in files:
        basecalls_file = os.path.join(input_dir, f, "basecalls.fastq")
        fasta_file = os.path.join(goldman_dir, "fastq", f + ".fasta")

        try:
            payload_accuracy, ground_truth, decoded = get_payload_accuracy(
                basecalls_file, fasta_file
            )
        except Exception as e:
            print(f"Warning: payload accuracy failed for {basecalls_file}: {e}")
            payload_accuracy = 0.0
            ground_truth = ""
            decoded = ""

        with open(os.path.join(input_dir, f, "payload_accuracy.txt"), "w") as fh:
            fh.write(str(payload_accuracy))
        with open(os.path.join(input_dir, f, "ground_truth_payload.txt"), "w") as fh:
            fh.write(ground_truth)
        with open(os.path.join(input_dir, f, "decoded_payload.txt"), "w") as fh:
            fh.write(decoded)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="input dir for synthetic data basecalls",
    )
    parser.add_argument(
        "--goldman-dir",
        type=str,
        required=True,
        help="goldman dir containing synthetic data",
    )

    args = parser.parse_args()
    main(input_dir=args.input_dir, goldman_dir=args.goldman_dir)