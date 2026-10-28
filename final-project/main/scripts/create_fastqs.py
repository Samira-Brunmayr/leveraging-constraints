"""Create fastq files for wick data from read_references.fasta for each genome"""

import argparse
import os

from constants import TEST_DATA, TRAIN_DATA
from ont_fast5_api.fast5_interface import get_fast5_file


def main(wick_datapath: str, verbose: bool = False):
    folders = [
        os.path.join(wick_datapath, f)
        for f in os.listdir(wick_datapath)
        if f != "genomes"
    ]

    if not os.path.isdir(os.path.join(wick_datapath, "genomes")):
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
            # in this case we make a dummy fastq from read_references.fasta
            fast5_dir = os.path.join(directory, "fast5")
            fast5_files = []
            # list all files in the directory
            for f in os.listdir(fast5_dir):
                if (
                    os.path.isfile(os.path.join(fast5_dir, f))
                    and os.path.splitext(f)[1].lower() == ".fast5"
                ):
                    fast5_files.append(f)

            if verbose:
                print(f"Found {len(fast5_files)} fast5 files for {folder_name}")
                print("Processing read_references.fasta")

            fasta_file = os.path.join(directory, "read_references.fasta")
            fastq_file = os.path.join(directory, "read_references.fastq")

            create_fastq(fast5_dir, fasta_file, fastq_file)

        elif folder_name in TEST_DATA:
            # in this case we don't need to do anything!
            print("Test data, skipping...")

    print("Done")
    return


def create_fastq(fast5_dir: str, fasta_file: str, fastq_file: str) -> None:
    """Create a fastq file for a given fasta file and fast5 directory"""
    # get all fast5 files
    fast5_files = []
    for f in os.listdir(fast5_dir):
        if (
            os.path.isfile(os.path.join(fast5_dir, f))
            and os.path.splitext(f)[1].lower() == ".fast5"
        ):
            fast5_files.append(os.path.join(fast5_dir, f))

    # read all fast5 files and store their ids
    basename_to_read_id = {}

    for fast5_file in fast5_files:
        with get_fast5_file(fast5_file, mode="r") as f5:
            number_of_reads = 0

            for read in f5.get_reads():
                read_id = read.read_id
                f5_basename = os.path.splitext(os.path.basename(fast5_file))[0]
                basename_to_read_id[f5_basename] = read_id
                number_of_reads += 1

            if number_of_reads != 1:
                raise ValueError(
                    f"{fast5_file} must contain 1 read (found {number_of_reads})"
                )

    # iterate through fasta and create an entry in the fastq for each
    seen_basenames = set()
    with open(fasta_file, "r") as fasta, open(fastq_file, "w") as fastq:
        lines = fasta.readlines()
        for i in range(0, len(lines), 2):
            header = lines[i].strip()[1:]
            sequence = lines[i + 1].strip()
            new_header = (
                f"{basename_to_read_id[header]} {header}"
                if header in basename_to_read_id
                else header
            )

            seen_basenames.add(header)

            fastq.write("@" + new_header + "\n")  # change '>' to '@'
            fastq.write(sequence + "\n")  # write sequence
            fastq.write("+\n")  # separator line
            fastq.write("I" * len(sequence) + "\n")  # dummy quality scores

    # check if any fast5 files don't have an entry in the fasta/fastq
    leftover_basenames = set(basename_to_read_id.keys()).difference(seen_basenames)

    if len(leftover_basenames) > 0:
        raise ValueError(
            f"No fasta entries for the following reads: {leftover_basenames}"
        )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script will process R. Wick's data by creating reference fastqs for all training data for future annotation with tombo"
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Directory with R. Wick's data"
    )
    parser.add_argument("--verbose", action="store_true", help="Toggle verbosity")

    args = parser.parse_args()
    main(args.path, args.verbose)
