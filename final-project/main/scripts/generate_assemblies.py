"""Call flye to generate assemblies given fastq"""

import argparse
import os
import subprocess

from main.scripts.constants import TEST_DATA, TRAIN_DATA


# Generate assemblies using Flye
def flye(
    fastq_path,  # str
    genome_size,  # str
    output_dir,  # str
    threads,  # int
    asm_coverage,  # Optional[int]
):
    command = [
        "flye",
        "--nano-raw",
        fastq_path,
        "--out-dir",
        output_dir,
        "--threads",
        str(threads),
        "--genome-size",
        genome_size,
    ]
    if asm_coverage is not None:
        command += ["--asm-coverage", str(asm_coverage)]
    process = subprocess.Popen(command)
    process.communicate()


def get_genome_size_from_fasta(fasta_path):
    """Find the genome size by counting bases in FASTA"""
    size = 0
    with open(fasta_path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                size += len(line.strip())
    return size


def main(
    path,  # str
    threads,  # int
    verbose,  # bool
    is_wick,  # bool
    output_dir=None,  # Optional[str]
    genome_size=None,  # Optional[str]
    genome_path=None,  # Optional[str]
    asm_coverage=None,  # Optional[int]
):
    # check either genome size or genome path are set
    if genome_path is None and genome_size is None:
        raise ValueError("Please ensure --genome-path or --genome-size are set")
    if genome_path is not None and genome_size is not None:
        raise ValueError("Please set only one of --genome-path or --genome-size")

    if not is_wick:
        if output_dir is None:
            raise ValueError("Please set --output-dir")

        # if genome path is set then let's calculate the genome size
        if genome_path is not None:
            genome_size = str(get_genome_size_from_fasta(genome_path))
            if verbose:
                print("Computed genome size of {}".format(genome_size))

        if verbose:
            print("Running Flye")
        flye(
            fastq_path=path,
            genome_size=genome_size,
            output_dir=output_dir,
            threads=threads,
            asm_coverage=asm_coverage,
        )

    if is_wick:
        if genome_path is None:
            raise ValueError(
                "--genome-path must be set with --is-wick to root wick data directory"
            )
        if output_dir is not None:
            raise ValueError("--output-dir must not be used with --is-wick")
        # if is_wick then --genome-path should be wick data root dir,
        # --path is root of folder created by basecall script
        # output dir not needed
        folders = [
            os.path.join(genome_path, f)
            for f in os.listdir(genome_path)
            if f != "genomes"
        ]

        if not os.path.isdir(os.path.join(genome_path, "genomes")):
            raise IOError("Expected genomes directory in root")

        genome_dir = os.path.join(genome_path, "genomes")

        for directory in folders:
            folder_name = os.path.basename(directory)

            if verbose:
                print("\nProcessing {}".format(folder_name))

            # check we have a fast5 directory
            if not os.path.isdir(os.path.join(directory, "fast5")):
                raise IOError(
                    "Couldn't locate fast5 directory for {}".format(folder_name)
                )

            # if train then make sure we have a read_references.fasta
            if (
                not os.path.isfile(os.path.join(directory, "read_references.fasta"))
                and folder_name in TRAIN_DATA
            ):
                raise IOError(
                    "Couldn't locate read_references.fasta for {}".format(folder_name)
                )

            # if test then make sure we have a fastq folder
            if (
                not os.path.isdir(os.path.join(directory, "fastq"))
                and folder_name in TEST_DATA
            ):
                raise IOError(
                    "Couldn't locate fastq directory for {}".format(folder_name)
                )

            # if test then make sure we have a basecalls.fastq
            fastq_directory = os.path.join(directory, "fastq")
            if (
                not os.path.isfile(os.path.join(fastq_directory, "basecalls.fastq"))
                and folder_name in TEST_DATA
            ):
                raise IOError(
                    "Couldn't locate basecalls.fastq for {}".format(folder_name)
                )

            if folder_name in TRAIN_DATA:
                if verbose:
                    print("Train data, skipping...")
            elif folder_name in TEST_DATA:
                genome_fasta = os.path.join(genome_dir, folder_name + ".fna")
                prediction_folder = os.path.join(path, folder_name)

                genome_size = str(get_genome_size_from_fasta(genome_fasta))
                if verbose:
                    print("Computed genome size of {}".format(genome_size))

                flye(
                    fastq_path=os.path.join(prediction_folder, "basecalls.fastq"),
                    genome_size=genome_size,
                    output_dir=os.path.join(prediction_folder, "flye_out"),
                    threads=threads,
                    asm_coverage=asm_coverage,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to basecalled fastq or to root dir if --is-wick",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Number of threads to use for Flye",
    )
    parser.add_argument(
        "--asm-coverage",
        type=int,
        help="asm coverage for Flye",
    )
    parser.add_argument("--verbose", action="store_true", help="Toggle verbosity")
    parser.add_argument(
        "--is-wick",
        action="store_true",
        help="If --path points to a wick structured basecalled root dir rather than a fastq",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for Flye if not --is-wick",
    )
    parser.add_argument(
        "--genome-size",
        type=str,
        help="Genome size for Flye, set this or --genome-path",
    )
    parser.add_argument(
        "--genome-path",
        type=str,
        help="Path to genome FASTA for Flye (or to wick data root dir if --is-wick), set this or --genome-size",
    )

    parser.add_argument

    args = parser.parse_args()
    main(
        path=args.path,
        threads=args.threads,
        asm_coverage=args.asm_coverage,
        verbose=args.verbose,
        is_wick=args.is_wick,
        output_dir=args.output_dir,
        genome_size=args.genome_size,
        genome_path=args.genome_path,
    )
