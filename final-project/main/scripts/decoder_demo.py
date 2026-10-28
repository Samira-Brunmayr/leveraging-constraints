"""Given a file, it will encode to DNA, simulate the fast5, basecall, and reconstruct the encoded payload"""

import os
import subprocess
from argparse import ArgumentParser
from typing import Any

from Bio import SeqIO

from main.dataset_processing.base_fast5_dataset import BaseFast5Dataset
from main.file_encoder import get_bytes
from main.models.lora.modules import LoRAOptions
from main.models.metrics import get_payload_accuracy
from main.scripts.dataset_generator import fix_read_ids, get_fragments, write_fastq
from main.scripts.utils import basecall, get_model


def main(
    input_file: str,
    path_to_code: str,
    output_folder: str,
    checkpoint: str,
    decoder: str,
    decoder_options: dict[str, Any],
    model_type: str,
    model_options: dict[str, Any],
):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    bs = get_bytes(input_file)
    fragments = get_fragments("goldman", bs, path_to_code)

    print("Writing fastq...")
    write_fastq(
        fragments,
        file_name,
        fastq_file := os.path.join(output_folder, file_name + ".fastq"),
    )
    print("Writing fasta")
    SeqIO.convert(
        fastq_file,
        "fastq",
        fasta_file := os.path.join(output_folder, file_name + ".fasta"),
        "fasta",
    )

    print("Squigulating...")
    pore = "dna-r9-min"
    if not os.path.isdir(blow5_folder := os.path.join(output_folder, "blow5")):
        os.makedirs(os.path.join(output_folder, "blow5"))
    subprocess.run(
        [
            "./tools/squigulator",
            fastq_file,
            f"-x{pore}",
            f"-o{os.path.join(blow5_folder, f'{file_name}.blow5')}",
            "--full-contigs",
            "--seed",
            "42",
            "--dwell-mean",
            str(13),
        ]
    )

    print("Converting to fast5...")
    fast5_dir = os.path.join(output_folder, "fast5")
    subprocess.run(["./tools/slow5tools", "s2f", blow5_folder, f"-d{fast5_dir}"])
    fix_read_ids(fast5_file := os.path.join(fast5_dir, file_name + ".fast5"))

    print("Basecalling...")
    model, _, _ = get_model(
        model_type=model_type,
        training=False,
        checkpoint=checkpoint,
        model_options=model_options,
    )
    fast5_dataset = BaseFast5Dataset(
        fast5_list=[fast5_file],
        buffer_size=1,
        window_size=2000,
        window_overlap=0,
    )

    read_stats = basecall(
        model=model,
        dataset=fast5_dataset,
        output_file=(basecalls_path := os.path.join(output_folder, "basecalls.fastq")),
        batch_size=64,
        beam_size=3,
        beam_threshold=0.0,
        verbose=True,
        overwrite=False,
        decoder=decoder,
        decoder_options=decoder_options,
    )
    succeeded_reads = read_stats["Read Succeeded"].sum()
    failed_reads = len(read_stats) - succeeded_reads
    print(
        f"{failed_reads} failed basecalls, and {succeeded_reads} successful basescalls"
    )
    read_stats.to_csv(
        os.path.join(output_folder, "basecall_stats.csv"),
        mode="w",
        header=True,
        index=False,
    )

    payload_accuracy, ground_truth, decoded_payload = get_payload_accuracy(
        basecalls_file=basecalls_path, fasta_file=fasta_file
    )
    print(f"Payload accuracy: {payload_accuracy}")
    with open(os.path.join(output_folder, "payload_accuracy.txt"), "w") as fh:
        fh.write(str(payload_accuracy))
    with open(os.path.join(output_folder, "ground_truth_payload.txt"), "w") as fh:
        fh.write(ground_truth)
    with open(os.path.join(output_folder, "decoded_payload.txt"), "w") as fh:
        fh.write(decoded_payload)
    return payload_accuracy


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="image file")
    parser.add_argument("--output-dir", type=str, required=True, help="output dir")
    parser.add_argument("--code", type=str, required=True, help="code file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="checkpoint file to load model weights",
        required=True,
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output files"
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
    model_options["lora_options"] = LoRAOptions(
        q=args.q,
        k=args.k,
        v=args.v,
        ffn=args.f,
        attn_out=args.attn_out,
        rank=args.lora_rank,
    )
    model_options["bitfit"] = False

    print(f"Decoder Options: {decoder_options}")
    print(f"Model Options: {model_options}")

    if args.model == "SACallLoRA":
        if (args.lora_rank is not None) and not (
            args.q or args.k or args.v or args.f or args.attn_out
        ):
            raise ValueError("Please specify lora options with -q -k -v -a -f")
    args = parser.parse_args()
    main(
        args.input_file,
        args.code,
        args.output_dir,
        checkpoint=args.checkpoint,
        decoder=args.decoder,
        decoder_options=decoder_options,
        model_type=args.model,
        model_options=model_options,
    )
