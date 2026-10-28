"""Generates a simulated dataset from fastq files"""

import argparse
import os
import re
import shutil
import subprocess
import uuid
from enum import StrEnum

import h5py

from main.dataset_processing.seeds import SQUIGULATOR_SEED

from ..file_encoder import (
    NucleotideString,
    base3_encode,
    fragments_from_base3,
    get_bytes,
    naive_encode,
)

"""Script that generates squiggles (fast5 format) for all files in a given directory"""


class PoreVersion(StrEnum):
    DNA_R9_MIN = "dna-r9-min"
    DNA_R9_PROM = "dna-r9-prom"
    DNA_R10_MIN = "dna-r10-min"
    DNA_R10_PROM = "dna-r10-prom"


def fix_read_ids(fast5_file: str):
    with h5py.File(fast5_file, "r+") as f:
        for read_group in f:
            if "Raw" in f[read_group]:
                raw_group = f[read_group]["Raw"]

                if "read_id" in raw_group.attrs:
                    old_read_id = raw_group.attrs["read_id"]
                    match = re.search(r"!([0-9a-fA-F\-]{36})!", str(old_read_id))
                    if match:
                        read_id = uuid.UUID(match.group(1))

                    raw_group.attrs.modify("read_id", str(read_id))


def write_fastq(
    fragments: list[NucleotideString], input_file_name: str, output_file: str
) -> None:
    """Write given fragments to fastq file"""
    output_dir = os.path.dirname(output_file)
    # create output directory if doesnt exist
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_file}", "w") as f:
        for i, frag in enumerate(fragments):
            id_ = str(uuid.uuid1())
            description = f"{input_file_name}_{i}"
            sequence = frag.string
            quality = "?" * len(sequence)

            lines = [
                "@" + id_ + " " + description + "\n",
                sequence + "\n",
                "+" + "\n",
                quality + "\n",
            ]
            f.writelines(lines)
    return


def write_genome(
    fragments: list[NucleotideString], input_file_name: str, output_file: str
):
    """Combine fragments into a 'genome' and build a fasta file"""
    output_dir = os.path.dirname(output_file)
    # create output directory if doesnt exist
    os.makedirs(output_dir, exist_ok=True)
    genome_str = "".join([frag.string for frag in fragments])

    with open(f"{output_file}", "w") as f:
        id_ = "genome"
        description = input_file_name
        sequence = genome_str

        lines = [
            ">" + id_ + " " + description + "\n",
            sequence + "\n",
        ]
        f.writelines(lines)
    return


def squigulate(
    input_dir: str,
    output_dir: str,
    pore_version: PoreVersion = PoreVersion.DNA_R9_MIN,
    seed: int = SQUIGULATOR_SEED,
) -> None:
    """Runs squigulator on all files in directory and saves outputs"""
    if not os.path.isdir(input_dir):
        raise ValueError(f"{input_dir} is not a directory")
    # get all file names
    files = [
        f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))
    ]

    # create output directory if doesnt exist
    os.makedirs(output_dir, exist_ok=True)
    for f in files:
        name, extension = os.path.splitext(f)
        if extension == ".fq" or extension == ".fastq":
            subprocess.run(
                [
                    "./tools/squigulator",
                    os.path.join(input_dir, f),
                    f"-x{pore_version.value}",
                    f"-o{os.path.join(output_dir, f'{name}.blow5')}",
                    "--full-contigs",
                    "--seed",
                    str(seed),
                    "--dwell-mean",
                    str(13)
                ], check=True
            )


def blow5_to_fast5(input_dir: str, output_dir: str) -> None:
    """Convert all blow5 files in directory to fast5 files"""
    if not os.path.isdir(input_dir):
        raise ValueError(f"{input_dir} is not a directory")
    # create output directory if doesnt exist
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["./tools/slow5tools", "s2f", input_dir, f"-d{output_dir}"], check=True)


def get_fragments(
    encoding_method: str, bs: bytearray, path_to_code: str | None
) -> list[NucleotideString]:
    if encoding_method == "goldman":
        return fragments_from_base3(base3_encode(bs, path_to_code))
    elif encoding_method == "naive":
        return naive_encode(bs, path_to_code)


def file_to_fastq(
    input_dir: str,
    fastq_dir: str,
    path_to_code: str,
    genome_dir: str,
    encoding_method: str,
) -> None:
    """Given input dir with files and output dir, turn all files into FASTQ"""
    if not os.path.isdir(input_dir):
        raise ValueError(f"{input_dir} is not a directory")
    # get all file names
    files = [
        f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))
    ]

    for f in files:
        name, _ = os.path.splitext(f)
        bs = get_bytes(os.path.join(input_dir, f))
        assert bs is not None
        fragments = get_fragments(encoding_method, bs, path_to_code)
        write_fastq(fragments, name, os.path.join(fastq_dir, name + ".fastq"))
        write_genome(fragments, name, os.path.join(genome_dir, name + ".fna"))


def multi_to_single_fast5(input_dir: str, output_dir: str, threads: int):
    subprocess.run(
        [
            "multi_to_single_fast5",
            "-i",
            input_dir,
            "-s",
            output_dir,
            "-t",
            str(threads)
        ], check=True
    )


def main(
    input_dir: str,
    output_dir: str,
    path_to_code: str,
    overwrite: bool,
    threads: int,
    single_fast5: bool,
    encoding_method: str,
    seed: int,
):
    if not os.path.isdir(input_dir):
        print(f"Error: The input directory {input_dir} does not exist.")
        return
    if not os.path.isfile(path_to_code):
        print(f"Error: The code file {path_to_code} does not exist.")
        return

    # make the output dir and then fastq/fast5 subdirs
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(fast5_dir := os.path.join(output_dir, "fast5")):
        os.makedirs(fast5_dir)

    if not os.path.isdir(fastq_dir := os.path.join(output_dir, "fastq")):
        os.makedirs(fastq_dir)

    if not os.path.isdir(genome_dir := os.path.join(output_dir, "genomes")):
        os.makedirs(genome_dir)

    if overwrite:
        # delete and recreate the fastq and fast5 dirs
        shutil.rmtree(fast5_dir)
        shutil.rmtree(fastq_dir)
        shutil.rmtree(genome_dir)
        os.makedirs(fast5_dir)
        os.makedirs(fastq_dir)
        os.makedirs(genome_dir)

    file_to_fastq(
        input_dir=input_dir,
        fastq_dir=fastq_dir,
        path_to_code=path_to_code,
        genome_dir=genome_dir,
        encoding_method=encoding_method,
    )
    temp_blow5 = os.path.join(output_dir, "temp_blow5")
    squigulate(
        input_dir=fastq_dir,
        output_dir=temp_blow5,
        seed=seed,
    )
    blow5_to_fast5(input_dir=temp_blow5, output_dir=fast5_dir)
    for f in os.listdir(fast5_dir):
        # first we want to fix the read ids in the fast5 file, then we move it
        fix_read_ids(os.path.join(fast5_dir, f))

        file_path = os.path.join(fast5_dir, f)
        file_name, file_ext = os.path.splitext(f)
        if not os.path.isdir(new_fast5_dir := os.path.join(fast5_dir, file_name)):
            os.makedirs(new_fast5_dir)
            shutil.move(file_path, os.path.join(new_fast5_dir, f))
            # now we want to turn it into single fast5 files (for tombo)
            if single_fast5:
                multi_to_single_fast5(
                    input_dir=new_fast5_dir, output_dir=new_fast5_dir, threads=threads
                )

                # now move all fast5 files up a directory
                single_files_dir = os.path.join(new_fast5_dir, "0")
                for f in os.listdir(single_files_dir):
                    full_path = os.path.join(single_files_dir, f)
                    shutil.move(full_path, new_fast5_dir)

                os.remove(os.path.join(new_fast5_dir, "filename_mapping.txt"))
                os.rmdir(single_files_dir)

                os.remove(os.path.join(new_fast5_dir, f"{file_name}.fast5"))

    shutil.rmtree(temp_blow5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts all FASTQ files in input directory into FAST5 in output directory"
    )

    parser.add_argument("--input-dir", type=str, help="The input directory")
    parser.add_argument("--output-dir", type=str, help="The output directory")
    parser.add_argument("--path-to-code", type=str, help="Path to prefix code")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrites fast5 directory"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads to use for converting multi fast5 to single fast5",
    )
    parser.add_argument(
        "--single-fast5",
        action="store_true",
        help="Convert multi fast5 to single fast5 files",
    )
    parser.add_argument(
        "--encoding-method", type=str, required=True, choices=["goldman", "naive"]
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SQUIGULATOR_SEED,
        help="Seed to pass to Squigulator",
    )
    args = parser.parse_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        path_to_code=args.path_to_code,
        overwrite=args.overwrite,
        threads=args.threads,
        single_fast5=args.single_fast5,
        encoding_method=args.encoding_method,
        seed=args.seed,
    )
