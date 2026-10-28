"""Adapted from

https://github.com/marcpaga/basecalling_architectures/blob/main/scripts/basecall_original.py
"""

import argparse
import gc
import os
from typing import Any, Literal

import torch

from main.dataset_processing.base_fast5_dataset import BaseFast5Dataset
from main.models.lora.modules import LoRAOptions
from main.scripts.utils import basecall, get_model

from .constants import TEST_DATA, TRAIN_DATA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    path: str,
    checkpoint: str,
    output: str,
    chunk_size: int,
    window_overlap: int,
    batch_size: int,
    beam_size: int,
    beam_threshold: float,
    is_wick: bool,
    is_synthetic: bool,
    verbose: bool,
    overwrite: bool,
    per_file: bool,
    decoder: Literal["greedy"] | Literal["beam_search"],
    decoder_options: dict[str, Any],
    model_type: Literal["SACall"],
    model_options: dict[str, Any],
):
    if is_wick and per_file:
        raise ValueError("--is-wick and --per-file are mutually exclusive")

    model, _, _ = get_model(
        model_type=model_type,
        training=False,
        checkpoint=checkpoint,
        model_options=model_options,
    )

    model = model.to(device)

    if per_file:
        # make output dir if doesnt exist
        name_suffix, _ = os.path.splitext(os.path.basename(output))
        output = os.path.dirname(output)
        if not os.path.isdir(output):
            os.makedirs(output)

        # get all fast5 files in input folder
        file_list = list()
        for f in os.listdir(path):
            if f.endswith(".fast5"):
                file_list.append(os.path.join(path, f))

        for fast5_file in file_list:
            fast5_dataset = BaseFast5Dataset(
                fast5_list=[fast5_file],
                buffer_size=1,
                window_size=chunk_size,
                window_overlap=window_overlap,
            )

            file_name, _ = os.path.splitext(os.path.basename(fast5_file))

            if verbose:
                print(f"Processing {file_name}...\n")

            read_stats = basecall(
                model=model,
                dataset=fast5_dataset,
                output_file=os.path.join(output, f"{file_name}_{name_suffix}.fastq"),
                batch_size=batch_size,
                beam_size=beam_size,
                beam_threshold=beam_threshold,
                verbose=True,
                overwrite=overwrite,
                decoder=decoder,
                decoder_options=decoder_options,
            )
            succeeded_reads = read_stats["Read Succeeded"].sum()
            failed_reads = len(read_stats) - succeeded_reads
            if verbose:
                print(
                    f"{failed_reads} failed basecalls, and {succeeded_reads} successful basescalls"
                )
            read_stats.to_csv(
                os.path.join(output, f"{file_name}_{name_suffix}_basecall_stats.csv"),
                mode="w",
                header=True,
                index=False,
            )
    elif is_wick:
        # create output folder if not exists
        if not os.path.isdir(output):
            os.makedirs(output)

        # go through each test directory and basecall it
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
                print("Training data, skipping...")

            elif folder_name in TEST_DATA:
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

                # if --is-wick then the output file should be a dir

                if not os.path.isdir(folder_path := os.path.join(output, folder_name)):
                    os.makedirs(folder_path)

                output_fastq = os.path.join(folder_path, "basecalls.fastq")

                fast5_dataset = BaseFast5Dataset(
                    fast5_list=fast5_files,
                    buffer_size=1,
                    window_size=chunk_size,
                    window_overlap=window_overlap,
                )

                read_stats = basecall(
                    model=model,
                    dataset=fast5_dataset,
                    output_file=output_fastq,
                    batch_size=batch_size,
                    beam_size=beam_size,
                    beam_threshold=beam_threshold,
                    verbose=True,
                    overwrite=overwrite,
                    decoder=decoder,
                    decoder_options=decoder_options,
                )

                succeeded_reads = read_stats["Read Succeeded"].sum()
                failed_reads = len(read_stats) - succeeded_reads
                if verbose:
                    print(
                        f"{failed_reads} failed basecalls, and {succeeded_reads} successful basescalls"
                    )
                read_stats.to_csv(
                    os.path.join(folder_path, "basecall_stats.csv"),
                    mode="w",
                    header=True,
                    index=False,
                )
            torch.cuda.empty_cache()
            gc.collect()
    elif is_synthetic:
        # create output folder if not exists
        if not os.path.isdir(output):
            os.makedirs(output)

        # get all file directories
        fast5_folder = os.path.join(path, "fast5")
        file_dirs = []
        for f in os.listdir(fast5_folder):
            if os.path.isdir(os.path.join(fast5_folder, f)):
                file_dirs.append(os.path.join(fast5_folder, f))

        # start basecalling
        for file_dir in file_dirs:
            if verbose:
                print(f"Processing {os.path.basename(file_dir)}...")

            # get fast 5 files for current directory
            fast5_files = []
            for f in os.listdir(file_dir):
                if os.path.splitext(f)[1] == ".fast5":
                    fast5_files.append(os.path.join(file_dir, f))

            if verbose:
                print(f"Found {len(fast5_files)} files to process")

            if not os.path.isdir(
                folder_path := os.path.join(output, os.path.basename(file_dir))
            ):
                os.makedirs(folder_path)

            output_fastq = os.path.join(folder_path, "basecalls.fastq")

            fast5_dataset = BaseFast5Dataset(
                fast5_list=fast5_files,
                buffer_size=1,
                window_size=chunk_size,
                window_overlap=window_overlap,
            )

            read_stats = basecall(
                model=model,
                dataset=fast5_dataset,
                output_file=output_fastq,
                batch_size=batch_size,
                beam_size=beam_size,
                beam_threshold=beam_threshold,
                verbose=True,
                overwrite=overwrite,
                decoder=decoder,
                decoder_options=decoder_options,
            )

            succeeded_reads = read_stats["Read Succeeded"].sum()
            failed_reads = len(read_stats) - succeeded_reads
            if verbose:
                print(
                    f"{failed_reads} failed basecalls, and {succeeded_reads} successful basescalls"
                )
            read_stats.to_csv(
                os.path.join(folder_path, "basecall_stats.csv"),
                mode="w",
                header=True,
                index=False,
            )
            torch.cuda.empty_cache()
            gc.collect()
    else:
        # get all fast5 files in input folder
        file_list = list()
        for f in os.listdir(path):
            if f.endswith(".fast5"):
                file_list.append(os.path.join(path, f))

        # if output file dir doesnt exist, create it
        if not os.path.isdir(os.path.dirname(output)):
            os.makedirs(os.path.dirname(output))

        fast5_dataset = BaseFast5Dataset(
            fast5_list=file_list,
            buffer_size=1,
            window_size=chunk_size,
            window_overlap=window_overlap,
        )

        read_stats = basecall(
            model=model,
            dataset=fast5_dataset,
            output_file=output,
            batch_size=batch_size,
            beam_size=beam_size,
            beam_threshold=beam_threshold,
            verbose=True,
            overwrite=overwrite,
            decoder=decoder,
            decoder_options=decoder_options,
        )

        succeeded_reads = read_stats["Read Succeeded"].sum()
        failed_reads = len(read_stats) - succeeded_reads
        if verbose:
            print(
                f"{failed_reads} failed basecalls, and {succeeded_reads} successful basescalls"
            )
        read_stats.to_csv(
            os.path.join(os.path.dirname(output), "basecall_stats.csv"),
            mode="w",
            header=True,
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="checkpoint file to load model weights",
        required=True,
    )
    parser.add_argument(
        "--output", type=str, help="output fastq file or directory", required=True
    )
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--window-overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--beam-threshold", type=float, default=0.0)
    parser.add_argument(
        "--is-wick", action="store_true", help="Use if --path refers to wick "
    )
    parser.add_argument(
        "--is-synthetic",
        action="store_true",
        help="Use if --path refers to synthetic data root ",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose flag")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output files"
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Flag for generating a basecalls.fastq file for each file in the fast5 dir, in this case --output-file adds the suffix to all files",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        choices=["beam_search", "greedy"],
        required=True,
        help="Decoding method to use",
    )
    parser.add_argument(
        "--homopolymer-penalty",
        type=float,
        help="Penalise beams with homopolymers exceeding this length",
        default=-1.0,
    )
    parser.add_argument(
        "--max-repeats",
        type=int,
        help="Max homopolymer length for beams that should be considered",
        default=-1,
    )
    parser.add_argument(
        "--desired-length",
        type=int,
        help="Desired length of output sequences",
        default=0,
    )
    parser.add_argument(
        "--length-lambda",
        type=float,
        help="Scaling for absolute length penalty",
        default=-1,
    )
    parser.add_argument(
        "--scale-timestep",
        action="store_true",
        help="Whether to scale the absolute length penalty by timestep",
        default=False,
    )
    parser.add_argument(
        "--discard-expected-length",
        type=float,
        help="Discard any beams that differ from the expected length fraction by this much",
        default=-1.0,
    )
    parser.add_argument(
        "--kl-lambda",
        type=float,
        help="Scaling for absolute length penalty",
        default=0.0,
    )
    parser.add_argument(
        "--discard-kl",
        type=float,
        help="Discard any beams that differ from the expected length fraction by this much",
        default=-1.0,
    )
    parser.add_argument(
        "--entropy-ratio",
        type=float,
        help="Entropy ratio of max ent",
        default=0.8,
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        choices=["SACall", "SACallLoRA"],
        help="what model to use",
    )
    parser.add_argument(
        "--decoder-finetune",
        action="store_true",
        default=False,
        help="Only finetune the decoder",
    )
    parser.add_argument(
        "--bitfit",
        action="store_true",
        help="Fine-tune all transformer encoder layer bias terms",
        default=False,
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        help="LoRA rank",
        default=None,
    )
    parser.add_argument(
        "-q",
        action="store_true",
        help="Enable LoRA on query projection",
    )
    parser.add_argument(
        "-k",
        action="store_true",
        help="Enable LoRA on key projection",
    )
    parser.add_argument(
        "-v",
        action="store_true",
        help="Enable LoRA on value projection",
    )
    parser.add_argument(
        "-a",
        action="store_true",
        dest="attn_out",
        help="Enable LoRA on attention output",
    )
    parser.add_argument(
        "-f",
        action="store_true",
        help="Enable LoRA on FFN layers",
    )
    args = parser.parse_args()

    decoder_options = {}
    decoder_options["homopolymer_penalty"] = args.homopolymer_penalty
    decoder_options["max_repeats"] = args.max_repeats
    decoder_options["length"] = args.desired_length
    decoder_options["length_lambda"] = args.length_lambda
    decoder_options["scale_by_timestep"] = args.scale_timestep
    decoder_options["discard_by_expected_length"] = args.discard_expected_length
    decoder_options["kl_lambda"] = args.kl_lambda
    decoder_options["discard_by_kl"] = args.discard_kl
    decoder_options["entropy_ratio"] = args.entropy_ratio

    model_options = {}
    model_options["decoder_finetune"] = args.decoder_finetune
    model_options["bitfit"] = args.bitfit
    model_options["lora_options"] = LoRAOptions(
        q=args.q,
        k=args.k,
        v=args.v,
        ffn=args.f,
        attn_out=args.attn_out,
        rank=args.lora_rank,
    )

    if args.verbose:
        print(f"Decoder Options: {decoder_options}")
        print(f"Model Options: {model_options}")

    if args.is_wick and args.is_synthetic:
        raise ValueError("--is-wick and --is-synthetic are mutually exclusive")

    if args.model == "SACallLoRA":
        if (args.lora_rank is not None) and not (
            args.q or args.k or args.v or args.f or args.attn_out
        ):
            raise ValueError("Please specify lora options with -q -k -v -a -f")
    main(
        path=args.path,
        checkpoint=args.checkpoint,
        output=args.output,
        chunk_size=args.chunk_size,
        window_overlap=args.window_overlap,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        beam_threshold=args.beam_threshold,
        is_wick=args.is_wick,
        is_synthetic=args.is_synthetic,
        verbose=args.verbose,
        overwrite=args.overwrite,
        per_file=args.per_file,
        decoder=args.decoder,
        decoder_options=decoder_options,
        model_type=args.model,
        model_options=model_options,
    )
