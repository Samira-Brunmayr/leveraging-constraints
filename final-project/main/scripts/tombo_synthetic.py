"""Run tombo on simulated data (annotating and then re-squiggling)"""

import argparse
import os
import re
import subprocess


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


def main(path, processes, verbose):
    """Takes path to synthetic rootdir and runs tombo on it"""
    root_dir = path
    fastq_dir = os.path.join(root_dir, "fastq")
    fast5_dir = os.path.join(root_dir, "fast5")
    genome_dir = os.path.join(root_dir, "genomes")

    if not os.path.isdir(root_dir):
        raise IOError("{} is not a directory".format(root_dir))

    if not os.path.isdir(fastq_dir):
        raise IOError("Could not find {}".format(fastq_dir))

    if not os.path.isdir(fast5_dir):
        raise IOError("Could not find {}".format(fast5_dir))

    if not os.path.isdir(genome_dir):
        raise IOError("Could not find {}".format(genome_dir))

    for f in os.listdir(fast5_dir):
        if not os.path.isdir(os.path.join(fast5_dir, f)):
            continue
        print(f)
        if verbose:
            print("Processing {}".format(f))
        preprocess_command = [
            "tombo",
            "preprocess",
            "annotate_raw_with_fastqs",
            "--fast5-basedir",
            os.path.join(fast5_dir, f),
            "--fastq-filenames",
            os.path.join(fastq_dir, f + ".fastq"),
            "--overwrite",
        ]

        resquiggle_command = [
            "tombo",
            "resquiggle",
            os.path.join(fast5_dir, f),
            os.path.join(genome_dir, f + ".fna"),
            "--processes",
            str(processes),
            "--dna",
            "--num-most-common-errors",
            "5",
            "--ignore-read-locks",
            "--overwrite",
            "--failed-reads-filename",
            os.path.join(fast5_dir, f + "/failed_reads.txt"),
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

        delete_failed_reads(os.path.join(fast5_dir, f + "/failed_reads.txt"), verbose)

    for f in os.listdir(fast5_dir):
        name, ext = os.path.splitext(f)
        if ext.lower() == ".index":
            os.remove(os.path.join(fast5_dir, f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path", type=str, required=True, help="Path to synthetic data root directory"
    )
    parser.add_argument(
        "--processes", type=int, default=2, help="Number of processes for tombo"
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    main(path=args.path, processes=args.processes, verbose=args.verbose)
