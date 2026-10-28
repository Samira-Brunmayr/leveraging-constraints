"""
Turns fast5 (tombo annotated) data into numpy arrays
"""

import argparse
import math
import multiprocessing as mp
import os
import random

import numpy as np
from tqdm import tqdm

from main.dataset_processing.normalization import normalize_signal_wrapper
from main.dataset_processing.read import read_fast5
from main.dataset_processing.seeds import DATA_PREPARE_READ_SHUFFLE

from .constants import TEST_DATA, TRAIN_DATA


def regular_break_points(n, chunk_len, overlap=0, align="mid"):
    """Define the start and end points of the raw data based on the
    window length and overlap

    Copied from: https://github.com/nanoporetech/bonito/blob/master/bonito/cli/convert.py

    Args:
        n (int): length of the raw data
        chunk_len (int): window size
        overlap (int): overlap between windows
        align (str): relative to the whole length, how should the windows align
    """
    num_chunks, remainder = divmod(n - overlap, chunk_len - overlap)
    start = {"left": 0, "mid": remainder // 2, "right": remainder}[align]
    starts = np.arange(
        start, start + num_chunks * (chunk_len - overlap), (chunk_len - overlap)
    )
    return np.vstack([starts, starts + chunk_len]).T


def segment_read(read_file, window_length, overlap, min_bases, max_bases):
    """Processes a single/multi fast5 read into chunks for deep learning training
    Adapted from https://github.com/marcpaga/basecalling_architectures/blob/main/scripts/data_prepare_numpy.py
    Args:
        read_file (str): fast5 file
        window_Length (int): size of the chunks in raw datapoints
        overlap (int): overlap between windows
        min_bases (int): minimum number of bases for a chunk to be considered
        max_bases (int): max number of bases for a chunk to be considered
    """
    if not os.path.isfile(read_file):
        print("File not found, skipping: " + read_file)
        return None, None

    multiread_obj = read_fast5(read_file)

    # lists to store all the segments
    x_list = list()
    y_list = list()

    for k in multiread_obj.keys():
        read_obj = multiread_obj[k]

        if read_obj.segmentation is None:
            continue

        read_data = dict()
        read_data["Dacs"] = read_obj.raw
        read_data["Ref_to_signal"] = (
            read_obj.segmentation["start"] + read_obj.start_rel_to_raw
        )
        read_data["Reference"] = "".join(read_obj.segmentation["base"])
        read_data["range"] = read_obj.range
        read_data["digitisation"] = read_obj.digitisation
        read_data["offset"] = read_obj.offset
        # read_id = read_obj.read_id

        # segment start and end points
        break_points = regular_break_points(
            len(read_data["Dacs"]), window_length, overlap=overlap, align="left"
        )

        # normalize the data
        sample = normalize_signal_wrapper(
            read_data["Dacs"],
            read_data["offset"],
            read_data["range"],
            read_data["digitisation"],
            method="noisiest",
            samples=100,
            threshold=6.0,
            factor=1.4826,
        )

        # get the positions of the sequence and the sequence as a whole
        pointers = read_data["Ref_to_signal"]
        reference_str = read_data["Reference"]
        # if sample is shorter than the window length, prepend 0s
        if len(sample) < window_length:
            sample = np.pad(sample, (window_length - len(sample), 0))
            x_list.append(sample)
            y_list.append(reference_str)
        else:
            for _, ((i, j), (ti, tj)) in enumerate(
                zip(break_points, np.searchsorted(pointers, break_points))
            ):
                # if they do not fit the min and max filters skip
                str_len = tj - ti
                if str_len > max_bases:
                    continue
                if str_len < min_bases:
                    continue

                # extract the data that we need
                x_list.append(sample[i:j])
                y_list.append(reference_str[ti:tj])

    if len(x_list) < 1:
        return None, None

    # stack all the arrays
    x = np.vstack(x_list)
    # for the y array we have to pre allocate the array as each
    # segment can be of a different length
    if max_bases == math.inf:
        max_len = window_length
    else:
        max_len = max_bases

    # use S1, do not use U1 as it is not supported by hdf5
    y = np.full((len(y_list), max_len), "", dtype="U1")
    for i, seg_str in enumerate(y_list):
        seg_str = list(seg_str)
        y[i, 0 : len(seg_str)] = seg_str

    return x, y


def segment_reads_and_write(
    read_files, output_file, window_length, overlap, min_bases, max_bases
):
    """Processes a set of fast5 reads into chunks for deep learning training
    and writes the chunks and labels into a numpy array
    Copied from https://github.com/marcpaga/basecalling_architectures/blob/main/scripts/data_prepare_numpy.py
    Args:
        read_file (str): fast5 file
        output_file (str): numpy file name
        window_Length (int): size of the chunks in raw datapoints
        overlap (int): overlap between windows
        min_bases (int): minimum number of bases for a chunk to be considered
        max_bases (int): max number of bases for a chunk to be considered
    """

    x_list = list()
    y_list = list()

    for read_file in read_files:
        x, y = segment_read(read_file, window_length, overlap, min_bases, max_bases)
        if x is not None:
            x_list.append(x)
            y_list.append(y)

    X = np.vstack(x_list)
    Y = np.vstack(y_list)

    np.savez(output_file, x=X, y=Y)


def main(
    path: str,
    out_dir: str,
    total_files: int,
    window_length: int,
    window_overlap: int,
    min_bases: int,
    max_bases: int,
    is_wick: bool,
    n_cores: int = 1,
    verbose: bool = True,
):
    """Processes a set of fast5 reads into chunks for deep learning training
    Adapted from https://github.com/marcpaga/basecalling_architectures/blob/main/scripts/data_prepare_numpy.py
    Args:
        wick_datapath (str): directory of wick root
        output_dir (str): dir where to store the processed data
        total_files (int): total number of files to divide the reads into
        window_Length (int): size of the chunks in raw datapoints
        window_overlap (int): overlap between windows
        min_bases (int): minimum number of bases for a chunk to be considered
        max_bases (int): max number of bases for a chunk to be considered
        n_cores (int): number of processes
        verbose (bool): output a progress bar
    """

    if verbose:
        print("Finding files to process")
    # find all the files that we have to process

    if is_wick:
        folders = [os.path.join(path, f) for f in os.listdir(path) if f != "genomes"]

        if not os.path.isdir(os.path.join(path, "genomes")):
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
                # find fast5 files
                fast5_files = []
                fast5_dir = os.path.join(directory, "fast5")
                for f in os.listdir(fast5_dir):
                    if (
                        os.path.isfile(os.path.join(fast5_dir, f))
                        and os.path.splitext(f)[1].lower() == ".fast5"
                    ):
                        file_path = os.path.join(fast5_dir, f)
                        fast5_files.append(file_path)

                if verbose:
                    print(f"Found {len(fast5_files)} fast5 files to process")

                # create output dir
                output_dir = os.path.join(directory, out_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                pool = mp.Pool(
                    processes=n_cores
                )  # add two to ensure that there is a process for the queue writer

                # divide files over total number of files
                random.seed(DATA_PREPARE_READ_SHUFFLE)
                random.shuffle(fast5_files)
                read_splits = np.array_split(fast5_files, total_files)

                jobs = list()
                for i, read_set in enumerate(read_splits):
                    output_file = os.path.join(output_dir, "data_" + str(i) + ".npz")
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
            elif folder_name in TEST_DATA:
                print("Test data, skipping...")

    elif not is_wick:
        fast5_files = []
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} is not a directory")
        for f in os.listdir(path):
            if (
                os.path.isfile(os.path.join(path, f))
                and os.path.splitext(f)[1].lower() == ".fast5"
            ):
                file_path = os.path.join(path, f)
                fast5_files.append(file_path)

        if verbose:
            print(f"Found {len(fast5_files)} fast5 files to process")

        # create output dir
        if not os.path.exists(out_dir):
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
        "--out-dir",
        type=str,
        help="Out directory for .npz files or folder name if --is-wick",
        required=True,
    )
    parser.add_argument(
        "--is-wick",
        action="store_true",
        help="Flag if --path is root R. Wick directory",
    )
    args = parser.parse_args()

    main(
        path=args.path,
        out_dir=args.out_dir,
        total_files=args.total_files,
        window_length=args.window_size,
        window_overlap=args.window_slide,
        min_bases=args.min_bases,
        max_bases=args.max_bases,
        n_cores=args.n_cores,
        verbose=args.verbose,
        is_wick=args.is_wick,
    )
