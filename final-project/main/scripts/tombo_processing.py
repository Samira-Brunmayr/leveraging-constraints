from __future__ import print_function  # python 2 compatibility for print()

import argparse
import os
import re
import subprocess

from constants import TEST_DATA, TRAIN_DATA

"""Run tombo on wick data (annotating and then re-squiggling)"""


def main(wick_datapath, verbose=False, processes=2):
    folders = [
        os.path.join(wick_datapath, f)
        for f in os.listdir(wick_datapath)
        if f != "genomes"
    ]

    if not os.path.isdir(os.path.join(wick_datapath, "genomes")):
        raise IOError("Expected genomes directory in root")

    genome_dir = os.path.join(wick_datapath, "genomes")

    for directory in folders:
        folder_name = os.path.basename(directory)

        if verbose:
            print("\nProcessing {}".format(folder_name))

        if not os.path.isdir(os.path.join(directory, "fast5")):
            raise IOError("Couldn't locate fast5 directory for {}".format(folder_name))

        if (
            not os.path.isfile(os.path.join(directory, "read_references.fasta"))
            and folder_name in TRAIN_DATA
        ):
            raise IOError(
                "Couldn't locate read_references.fasta for {}".format(folder_name)
            )

        if (
            not os.path.isdir(os.path.join(directory, "fastq"))
            and folder_name in TEST_DATA
        ):
            raise IOError("Couldn't locate fastq directory for {}".format(folder_name))

        fastq_directory = os.path.join(directory, "fastq")
        if (
            not os.path.isfile(os.path.join(fastq_directory, "basecalls.fastq"))
            and folder_name in TEST_DATA
        ):
            raise IOError("Couldn't locate basecalls.fastq for {}".format(folder_name))

        if folder_name in TRAIN_DATA:
            fastq_file = os.path.join(directory, "read_references.fastq")
            fast5_dir = os.path.join(directory, "fast5")
            reference_genome_name = folder_name.split("-")[0]
            genome = os.path.join(genome_dir, "{}.fna".format(reference_genome_name))
            failed_reads_file = os.path.join(directory, "failed_reads.txt")

            preprocess_command = [
                "tombo",
                "preprocess",
                "annotate_raw_with_fastqs",
                "--fast5-basedir",
                fast5_dir,
                "--fastq-filenames",
                fastq_file,
            ]

            resquiggle_command = [
                "tombo",
                "resquiggle",
                fast5_dir,
                genome,
                "--processes",
                str(processes),
                "--dna",
                "--num-most-common-errors",
                "5",
                "--ignore-read-locks",
                "--overwrite",
                "--failed-reads-filename",
                failed_reads_file,
            ]

            if verbose:
                print("Beginning annotation...")
            process = subprocess.Popen(preprocess_command)
            process.communicate()

            if verbose:
                print("Beginning resquiggling...")

            process = subprocess.Popen(resquiggle_command)
            process.communicate()

            if verbose:
                print("Deleting failed reads...")

            delete_failed_reads(failed_reads_file, verbose)

        elif folder_name in TEST_DATA:
            print("Test data, skipping...")


def delete_failed_reads(failed_reads_file, verbose=False):
    """Delete all fast5 files that failed to read"""
    with open(failed_reads_file, "r") as f:
        content = f.read()
        failed_files = re.findall(r"BaseCalled_template:::(\S+?)(?:,|\s|$)", content)

        if verbose:
            print("Removing {} reads".format(len(failed_files)))

        for failed_file in failed_files:
            if os.path.exists(failed_file):
                os.remove(failed_file)
            else:
                raise IOError("Couldn't remove {}".format(failed_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script processes R. Wick's data by creating reference fastqs for all training data for future annotation with tombo"
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Directory with R. Wick's data"
    )
    parser.add_argument("--verbose", action="store_true", help="Toggle verbosity")
    parser.add_argument(
        "--processes",
        type=int,
        default=2,
        help="Number of processes to use for resquiggling",
    )

    args = parser.parse_args()
    main(args.path, args.verbose, args.processes)
