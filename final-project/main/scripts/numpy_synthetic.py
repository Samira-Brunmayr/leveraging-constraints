"""Creates numpy arrays from simulated fast5 files
Adapted from https://github.com/marcpaga/basecalling_architectures/blob/main/scripts/data_prepare_numpy.py
"""

import argparse
import math
import multiprocessing as mp
import os
import random
import shutil

import numpy as np
from tqdm import tqdm

from main.dataset_processing.seeds import DATA_PREPARE_READ_SHUFFLE
from main.scripts.data_prepare_numpy import segment_reads_and_write


def main(
    path: str,
    output_dirname: str,
    total_files: int,
    window_length: int,
    window_overlap: int,
    min_bases: int,
    max_bases: int,
    n_cores: int = 1,
    verbose: bool = True,
    overwrite: bool = False,
):
    """Processes a set of fast5 reads into chunks for deep learning training

    Args:
        path (str): directory of synthetic data
        output_dir (str): dir where to store the processed data
        total_files (int): total number of files to divide the reads into
        window_Length (int): size of the chunks in raw datapoints
        window_overlap (int): overlap between windows
        min_bases (int): minimum number of bases for a chunk to be considered
        max_bases (int): max number of bases for a chunk to be considered
        n_cores (int): number of processes
        verbose (bool): output a progress bar
    """
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

    if verbose:
        print("Finding files to process")
    # find all the files that we have to process

    for f in os.listdir(fast5_dir):
        fast5_files = []
        for fast5_file in os.listdir(file_fast5_dir := os.path.join(fast5_dir, f)):
            if (
                os.path.isfile(os.path.join(file_fast5_dir, fast5_file))
                and os.path.splitext(fast5_file)[1].lower() == ".fast5"
            ):
                file_path = os.path.join(file_fast5_dir, fast5_file)
                fast5_files.append(file_path)

        if verbose:
            print(f"Found {len(fast5_files)} fast5 files to process")

        # create output dir
        out_dir = os.path.join(file_fast5_dir, output_dirname)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        else:
            if overwrite:
                shutil.rmtree(out_dir)
                os.makedirs(out_dir)

        pool = mp.Pool(
            processes=n_cores
        )  # add two to ensure that there is a process for the queue writer

        # divide files over total number of files
        random.seed(DATA_PREPARE_READ_SHUFFLE)
        random.shuffle(fast5_files)
        read_splits = np.array_split(fast5_files, total_files)

        jobs = list()
        for i, read_set in enumerate(read_splits):
            output_file = os.path.join(out_dir, "data_" + str(i) + ".npz")
            jobs.append(
                pool.apply_async(
                    segment_reads_and_write,
                    (
                        read_set.tolist(),
                        output_file,
                        window_length,
                        window_overlap,
                        min_bases,
                        max_bases,
                    ),
                )
            )
        if verbose:
            print("Processing " + str(len(jobs)) + " files")
        for job in tqdm(jobs, disable=not verbose):
            job.get()

        pool.close()
        pool.join()

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total-files", type=int, help="Total number of files to divide the reads into"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        help="Size of the window of a raw data segment",
        default=2000,
    )
    parser.add_argument(
        "--window-slide",
        type=int,
        help="Number of datapoints of overlap between sequential segments",
        default=0,
    )
    parser.add_argument(
        "--min-bases",
        type=int,
        help="Minimum number of bases for a segment to be kept",
        default=10,
    )
    parser.add_argument(
        "--max-bases",
        type=int,
        help="Maximum number of bases for a segment to be kept",
        default=math.inf,
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        help="Number of parallel processes, do not use more processes than total files as it will be wasted resources",
        default=1,
    )
    parser.add_argument("--verbose", action="store_true", help="Print a progress bar")
    parser.add_argument(
        "--path",
        type=str,
        help="Location of the fast5 files",
        required=True,
    )
    parser.add_argument(
        "--output-dirname",
        type=str,
        help="Out directory name for .npz files",
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite directories",
    )
    args = parser.parse_args()

    main(
        path=args.path,
        output_dirname=args.output_dirname,
        total_files=args.total_files,
        window_length=args.window_size,
        window_overlap=args.window_slide,
        min_bases=args.min_bases,
        max_bases=args.max_bases,
        n_cores=args.n_cores,
        verbose=args.verbose,
        overwrite=args.overwrite,
    )
