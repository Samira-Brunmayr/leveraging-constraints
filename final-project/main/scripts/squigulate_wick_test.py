import argparse
import os
import re
import shutil
import subprocess
import uuid
from enum import StrEnum

import h5py

from main.dataset_processing.seeds import SQUIGULATOR_SEED
from main.scripts.constants import TEST_DATA

from ..file_encoder import (
    NucleotideString,
)

"""Simulates the wick test data"""


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


def squigulate(
    input_dir: str,
    output_dir: str,
    pore_version: PoreVersion = PoreVersion.DNA_R9_MIN,
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
                    str(SQUIGULATOR_SEED),
                    "--dwell-mean",
                    str(13),
                ]
            )


def blow5_to_fast5(input_dir: str, output_dir: str) -> None:
    """Convert all blow5 files in directory to fast5 files"""
    if not os.path.isdir(input_dir):
        raise ValueError(f"{input_dir} is not a directory")
    # create output directory if doesnt exist
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["./tools/slow5tools", "s2f", input_dir, f"-d{output_dir}"])


def main(input_dir: str, output_dir: str):
    if not os.path.isdir(input_dir):
        print(f"Error: The input directory {input_dir} does not exist.")
        return

    # make the output dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # make the genomes directory if it doesnt exist
    shutil.copytree(
        os.path.join(
            input_dir,
            "genomes",
        ),
        os.path.join(output_dir, "genomes"),
    )

    for genome in os.listdir(input_dir):
        full_genome_path = os.path.join(input_dir, genome)
        if genome in TEST_DATA:
            print(f"Processing {genome}...")
            # create output folder, fast5 folder and fastq folder
            if not os.path.isdir(out_genome_dir := os.path.join(output_dir, genome)):
                os.makedirs(out_genome_dir)
            if not os.path.isdir(
                out_genome_fast5 := os.path.join(out_genome_dir, "fast5")
            ):
                os.makedirs(out_genome_fast5)
            if not os.path.isdir(
                out_genome_fastq := os.path.join(out_genome_dir, "fastq")
            ):
                os.makedirs(out_genome_fastq)

            # copy basecalls to fast q folder
            basecalls = os.path.join(full_genome_path, "fastq", "basecalls.fastq")
            shutil.copy(basecalls, os.path.join(out_genome_fastq, "basecalls.fastq"))
            squigulate(
                input_dir=out_genome_fastq,
                output_dir=(temp_blow5 := os.path.join(out_genome_dir, "temp_blow5")),
            )

            blow5_to_fast5(input_dir=temp_blow5, output_dir=out_genome_fast5)
            for f in os.listdir(out_genome_fast5):
                # first we want to fix the read ids in the fast5 file, then we move it
                fix_read_ids(os.path.join(out_genome_fast5, f))

            shutil.rmtree(temp_blow5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script squigulates the wick test data"
    )

    parser.add_argument("--path", type=str, help="Path to wick root directory")
    parser.add_argument("--output-dir", type=str, help="The output directory")
    args = parser.parse_args()
    main(input_dir=args.path, output_dir=args.output_dir)
