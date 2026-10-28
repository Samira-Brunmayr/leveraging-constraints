"""Given a reference genome and a consensus assembly, compute the consensus accuracy"""

import argparse
import os
from typing import Optional

import pandas as pd

from main.models.metrics import get_consensus_accuracy
from main.scripts.constants import TEST_DATA, TRAIN_DATA


def save_consensus_accuracy(
    genome_path: str,
    assembly_path: str,
    chunk_size: int,
    output_dir: str,
    verbose: bool,
):
    consensus_accuracy, consensus_metrics = get_consensus_accuracy(
        genome_path, assembly_path, chunk_size
    )
    if verbose:
        print(f"Consensus accuracy: {consensus_accuracy}")

    consensus_metrics.to_csv(
        os.path.join(output_dir, "consensus_evaluation.csv"),
        mode="w",
        header=True,
        index=False,
    )

    pd.DataFrame([{"Consensus Accuracy": consensus_accuracy}]).to_csv(
        os.path.join(output_dir, "consensus_accuracy.csv"),
        mode="w",
        header=True,
        index=False,
    )


def main(
    genome_path: str,
    assembly_path: str,
    chunk_size: int,
    verbose: bool,
    is_wick: bool,
    output_dir: Optional[str],
):
    # if not is wick make sure output_file is provided
    if not is_wick and output_dir is None:
        raise ValueError("--output-dir must be provided if not --is-wick")

    if not is_wick:
        # create output folder
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        save_consensus_accuracy(
            genome_path=genome_path,
            assembly_path=assembly_path,
            chunk_size=chunk_size,
            output_dir=output_dir,
            verbose=verbose,
        )
    elif is_wick:
        if genome_path is None:
            raise ValueError(
                "--genome-path must be set with --is-wick to root wick data directory"
            )
        if output_dir is not None:
            raise ValueError("--output-dir must not be used with --is-wick")
        # if is_wick then --genome-path should be wick data root dir,
        # --assembly-path is root of folder created by basecall script
        # output dir not needed
        folders = [
            os.path.join(genome_path, f)
            for f in os.listdir(genome_path)
            if f != "genomes"
        ]

        if not os.path.isdir(os.path.join(genome_path, "genomes")):
            raise FileNotFoundError("Expected genomes directory in root")

        genome_dir = os.path.join(genome_path, "genomes")

        for directory in folders:
            folder_name = os.path.basename(directory)

            if verbose:
                print("\nProcessing {}".format(folder_name))

            # check we have a fast5 directory
            if not os.path.isdir(os.path.join(directory, "fast5")):
                raise FileNotFoundError(
                    "Couldn't locate fast5 directory for {}".format(folder_name)
                )

            # if train then make sure we have a read_references.fasta
            if (
                not os.path.isfile(os.path.join(directory, "read_references.fasta"))
                and folder_name in TRAIN_DATA
            ):
                raise FileNotFoundError(
                    "Couldn't locate read_references.fasta for {}".format(folder_name)
                )

            # if test then make sure we have a fastq folder
            if (
                not os.path.isdir(os.path.join(directory, "fastq"))
                and folder_name in TEST_DATA
            ):
                raise FileNotFoundError(
                    "Couldn't locate fastq directory for {}".format(folder_name)
                )

            # if test then make sure we have a basecalls.fastq
            fastq_directory = os.path.join(directory, "fastq")
            if (
                not os.path.isfile(os.path.join(fastq_directory, "basecalls.fastq"))
                and folder_name in TEST_DATA
            ):
                raise FileNotFoundError(
                    "Couldn't locate basecalls.fastq for {}".format(folder_name)
                )

            if folder_name in TRAIN_DATA:
                if verbose:
                    print("Train data, skipping...")
            elif folder_name in TEST_DATA:
                genome_fasta = os.path.join(genome_dir, folder_name + ".fna")
                prediction_folder = os.path.join(assembly_path, folder_name)

                if not os.path.isfile(
                    os.path.join(prediction_folder, "medaka_out/consensus.fasta")
                ):
                    raise FileNotFoundError(
                        f"Could not find consensus.fasta in assembly dir {folder_name}"
                    )

                save_consensus_accuracy(
                    genome_path=genome_fasta,
                    assembly_path=os.path.join(
                        prediction_folder, "medaka_out/consensus.fasta"
                    ),
                    chunk_size=chunk_size,
                    output_dir=os.path.join(prediction_folder, "evaluation"),
                    verbose=verbose,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--genome-path",
        type=str,
        required=True,
        help="Path to reference genome FASTA or to root wick data dir if --is-wick",
    )
    parser.add_argument(
        "--assembly-path",
        type=str,
        required=True,
        help="Path to consensus assembly FASTA or to root basecalled dir if --is-wick",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size to use when computing consensus accuracy",
    )
    parser.add_argument("--verbose", action="store_true", help="Toggle verbosity")
    parser.add_argument(
        "--is-wick",
        action="store_true",
        help="If --path points to a wick structured basecalled root dir rather than a fasta",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output file for .csv if not --is-wick",
    )

    parser.add_argument

    args = parser.parse_args()
    main(
        genome_path=args.genome_path,
        assembly_path=args.assembly_path,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
        is_wick=args.is_wick,
        output_dir=args.output_dir,
    )
