"""Evaluates the model basecall output (computes all metrics and saves to file)"""

import argparse
import os

import mappy
import pandas as pd
from Bio import SeqIO

from main.models.metrics import get_hits, get_metrics, to_fasta
from main.scripts.constants import TEST_DATA, TRAIN_DATA


def save_metrics(
    reference_path: str, prediction_path: str, output_dir: str, verbose: bool
) -> None:
    reference_base, reference_extension = os.path.splitext(reference_path)
    prediction_base, prediction_extension = os.path.splitext(prediction_path)

    if reference_extension not in [".fasta", ".fastq"]:
        raise FileExistsError(f"{reference_path} is not a FASTA or FASTQ file")

    if prediction_extension != ".fastq":
        raise FileExistsError(f"{prediction_path} is not a FASTQ file")

    if reference_extension == ".fastq":
        # convert to fasta
        fastq_path = reference_path
        reference_path = reference_base + ".fasta"
        to_fasta(fastq_path, reference_path)

    # load references with mappy
    # load references into an aligner
    if verbose:
        print("Creating mappy aligner")
    aln = mappy.Aligner(reference_path, preset="map-ont")
    if not aln:
        raise Exception("Failed to load/build minimap2 index")

    if verbose:
        print("Mapping predictions")
    # get primary alignment hits for each read in the prediction file
    prediction_records = SeqIO.parse(prediction_path, "fastq")
    primary_hits, failed_aligns = get_hits(aln, prediction_records, primary=True)
    if verbose:
        print(f"{len(primary_hits)} hits and {failed_aligns} failed alignments")
    if verbose:
        print("Computing metrics")
    metrics = get_metrics(primary_hits)

    # make output directory if it doesnt exist
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    metrics.to_csv(
        os.path.join(output_dir, "evaluation.csv"), mode="w", header=True, index=False
    )
    pd.DataFrame(
        [{"Aligned Reads": len(primary_hits), "Failed alignments": failed_aligns}]
    ).to_csv(
        os.path.join(output_dir, "alignment_stats.csv"),
        mode="w",
        header=True,
        index=False,
    )


def main(
    reference_path: str,
    prediction_path: str,
    is_wick: bool,
    is_synthetic: bool,
    verbose: bool,
    output_dir: str | None = None,
):
    if is_synthetic:
        if output_dir is not None:
            raise ValueError("--output-dir must not be used with --is-synthetic")
        # reference-path is the synthetic data root
        # prediction path is the folder created by basecall script

        fastq_dir = os.path.join(reference_path, "fastq")
        fastq_files = [
            f for f in os.listdir(fastq_dir) if os.path.splitext(f)[1] == ".fastq"
        ]
        for fastq_file in fastq_files:
            if verbose:
                print(f"\nProcessing {os.path.basename(fastq_file)}")
            # get basecalls
            basecalls_path = os.path.join(
                prediction_path, os.path.splitext(fastq_file)[0], "basecalls.fastq"
            )

            reference_seqs = os.path.join(reference_path, "fastq", fastq_file)

            if not os.path.isfile(basecalls_path) or not os.path.isfile(reference_seqs):
                continue

            print("Found predictions and references, evaluating...")
            save_metrics(
                reference_path=reference_seqs,
                prediction_path=basecalls_path,
                output_dir=os.path.join(
                    prediction_path, os.path.splitext(fastq_file)[0], "evaluation"
                ),
                verbose=verbose,
            )
    elif is_wick:
        if output_dir is not None:
            raise ValueError("--output-dir must not be used with --is-wick")
        # if is_wick then reference-path should be wick root dir,
        # predicion path is root of folder created by basecall script
        # output dir not needed
        folders = [
            os.path.join(reference_path, f)
            for f in os.listdir(reference_path)
            if f != "genomes"
        ]

        if not os.path.isdir(os.path.join(reference_path, "genomes")):
            raise FileNotFoundError("Expected genomes directory in root")

        for directory in folders:
            folder_name = os.path.basename(directory)

            if verbose:
                print(f"\nProcessing {folder_name}")

            # check we have a fast5 directory
            if not os.path.isdir(os.path.join(directory, "fast5")):
                raise FileNotFoundError(
                    f"Couldn't locate fast5 directory for {folder_name}"
                )

            # if train then make sure we have a read_references.fasta
            if (
                not os.path.isfile(os.path.join(directory, "read_references.fasta"))
                and folder_name in TRAIN_DATA
            ):
                raise FileNotFoundError(
                    f"Couldn't locate read_references.fasta for {folder_name}"
                )

            # if test then make sure we have a fastq folder
            if (
                not os.path.isdir(os.path.join(directory, "fastq"))
                and folder_name in TEST_DATA
            ):
                raise FileNotFoundError(
                    f"Couldn't locate fastq directory for {folder_name}"
                )

            # if test then make sure we have a basecalls.fastq
            fastq_directory = os.path.join(directory, "fastq")
            if (
                not os.path.isfile(os.path.join(fastq_directory, "basecalls.fastq"))
                and folder_name in TEST_DATA
            ):
                raise FileNotFoundError(
                    f"Couldn't locate basecalls.fastq for {folder_name}"
                )

            if folder_name in TRAIN_DATA:
                if verbose:
                    print("Train data, skipping...")
            elif folder_name in TEST_DATA:
                prediction_folder = os.path.join(prediction_path, folder_name)
                save_metrics(
                    reference_path=os.path.join(fastq_directory, "basecalls.fastq"),
                    prediction_path=os.path.join(prediction_folder, "basecalls.fastq"),
                    output_dir=os.path.join(prediction_folder, "evaluation"),
                    verbose=verbose,
                )
    else:
        if output_dir is None:
            raise ValueError("--output-dir must be set when not using --is-wick")
        save_metrics(reference_path, prediction_path, output_dir, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path where the data for the dataloaders is stored",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path where the data for the dataloaders is stored",
    )
    parser.add_argument("--output-dir", type=str, help="Path where the model is saved")
    parser.add_argument(
        "--is-wick",
        action="store_true",
        help="is data-dir the path to R. Wick's preprocessed data",
    )
    parser.add_argument(
        "--is-synthetic",
        action="store_true",
        help="is data-dir the path to R. Wick's preprocessed data",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="toggle verbosity",
    )
    args = parser.parse_args()

    if args.is_wick and args.is_synthetic:
        raise ValueError("--is-wick and --is-synthetic are mutually exclusive")

    main(
        reference_path=args.reference,
        prediction_path=args.predictions,
        output_dir=args.output_dir,
        is_wick=args.is_wick,
        is_synthetic=args.is_synthetic,
        verbose=args.verbose,
    )
