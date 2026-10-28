"""Script for training a model

Adapted from https://github.com/marcpaga/basecalling_architectures/blob/main/scripts/train_original.py
"""

import argparse
import functools
import json
import os
import time
from contextlib import nullcontext
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile

from main.dataset_processing.seeds import GLOBAL_SEED, set_seed
from main.models.lora.model import LoRAOptions

from .utils import (
    convert_counter_to_dict,
    create_dataset,
    evaluate,
    get_decoder,
    get_loss,
    get_model,
    save,
    trace_handler,
    train_step,
    validation_step,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALIDATE_EVERY = 500
CHECKPOINT_EVERY = 20000
CLIPPING_VALUE = 2

SKIP_FIRST = 10
WAIT = 5
WARMUP = 5
ACTIVE = 1
REPEAT = 10
TOTAL_PROFILER_STEPS = SKIP_FIRST + (WAIT + WARMUP + ACTIVE) * REPEAT


def main(
    data_dir: str,
    is_wick: bool,
    is_synthetic: bool,
    output_dir: str,
    decoder: Literal["beam_search"] | Literal["greedy"],
    loss: Literal["ctc"],
    num_epochs: int,
    batch_size: int,
    starting_lr: float,
    warmup_steps: int,
    is_profiling: bool,
    max_steps: int | None,
    model_options: dict[str, Any],
    model_type: Literal["SACall"] | Literal["SACallLoRA"],
    checkpoint: str | None = None,
    finetune_checkpoint: str | None = None,
    numpy_dir: str = "",
    image_start: int | None = None,
    image_end: int | None = None,
    seed: int = GLOBAL_SEED,
):
    # set seeds
    set_seed(seed)

    # Initialise dataset and dataloaders
    print("Creating dataset")
    dataset, loader_train, loader_validation = create_dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        is_wick=is_wick,
        is_synthetic=is_synthetic,
        numpy_dir=numpy_dir,
        image_start=image_start,
        image_end=image_end,
    )
    print(f"Found {batch_size * len(loader_train)} training samples")
    print(f"Found {batch_size * len(loader_validation)} validation samples")

    # Initialise model
    print("Creating model")
    model, optimizer, scheduler = get_model(
        model_type=model_type,
        training=True,
        starting_lr=starting_lr,
        dataset=dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        checkpoint=checkpoint,
        finetune_checkpoint=finetune_checkpoint,
        model_options=model_options,
    )

    print(f"lora options: {model_options['lora_options'].__dict__}")

    trainable_params = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    print("Trainable parameters:", trainable_params)

    # Initialise decoder and loss
    decode_fn = get_decoder(decoder)
    loss_fn = get_loss(loss)

    print("Creating outputs")
    # TODO overwrite arg i.e. delete and recreate
    # make outputdirs
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    # check output dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # create log
    log = pd.DataFrame(
        columns=[
            "epoch",
            "num_steps",
            "train_loss",
            "ema_train_loss",
            "val_loss",
            "ema_val_loss",
            "train_acc",
            "ema_train_acc",
            "val_acc",
            "ema_val_acc",
            "elapsed_time",
        ]
    )
    log.to_csv(os.path.join(output_dir, "log.csv"), index=False)

    # create profiler log
    if is_profiling:
        profiler_log = pd.DataFrame(
            columns=[
                "Name",
                "Count",
                "Total CPU Time",  # microseconds
                "Total CUDA Time",  # microseconds
                "Total CPU Usage",  # Bytes
                "Total CUDA Usage",  # Bytes
            ]
        )
        profiler_log.to_csv(os.path.join(output_dir, "profile.csv"), index=False)

    print("Training")
    total_num_steps = 1
    ema_train_loss = None
    ema_train_acc = None
    ema_val_loss = None
    ema_val_acc = None
    start_time = time.time()
    for epoch in range(num_epochs):
        if max_steps is not None and total_num_steps == max_steps:
            break
        # use this to restart the in case we finish all the validation data
        validation_iterator = iter(loader_validation)
        # profile
        is_profiling = is_profiling and epoch == 0
        with (
            profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    skip_first=SKIP_FIRST,
                    wait=WAIT,
                    warmup=WARMUP,
                    active=ACTIVE,
                    repeat=REPEAT,
                ),
                on_trace_ready=functools.partial(
                    trace_handler, model_tags=model.profiler_tags, output_dir=output_dir
                ),
                profile_memory=True,
                record_shapes=True,
                with_flops=True,
            )
            if is_profiling
            else nullcontext() as prof
        ):
            # iterate over the train data
            for train_batch_num, train_batch in enumerate(loader_train):
                if max_steps is not None and total_num_steps == max_steps:
                    break
                train_loss, predictions, flops = train_step(
                    model=model,
                    batch=train_batch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    is_profiling=is_profiling,
                    count_flops=is_profiling
                    and train_batch_num
                    == 0,  # only count flops on the very first forward pass
                )
                if flops is not None:
                    # convert flops object to dict and write it to json file
                    flop_counts = convert_counter_to_dict(
                        flops.by_module_and_operator()
                    )
                    with open(os.path.join(output_dir, "flops.json"), "w") as f:
                        json.dump(flop_counts, f)
                total_num_steps += 1

                if train_batch_num < TOTAL_PROFILER_STEPS and is_profiling:
                    # step profiler
                    prof.step()
                # average profiler results
                if train_batch_num == TOTAL_PROFILER_STEPS and is_profiling:
                    df = pd.read_csv(os.path.join(output_dir, "profile.csv"))
                    df = df.groupby("Name", as_index=False).mean()
                    df.to_csv(
                        os.path.join(output_dir, "profile.csv"),
                        mode="w",
                        header=True,
                        index=False,
                    )

                if total_num_steps % VALIDATE_EVERY == 0:
                    elapsed_time = time.time() - start_time
                    # calculate accuracy for the training only here since doing for every batch
                    # is expensive and slow...
                    print(f"Step: {total_num_steps}")

                    predictions_decoded = decode_fn(predictions)
                    y = train_batch["y"].cpu().numpy()
                    y_references = dataset.encoded_array_to_list_strings(y)
                    metrics = evaluate(predictions_decoded, y_references)
                    train_acc = np.mean(metrics["metric.accuracy"])
                    print(f"Train metrics: {train_acc}")
                    try:
                        validation_batch = next(validation_iterator)
                    except StopIteration:
                        validation_iterator = iter(loader_validation)
                        validation_batch = next(validation_iterator)

                    # calculate and log the validation results
                    val_loss, predictions = validation_step(
                        model=model,
                        batch=validation_batch,
                        loss_fn=loss_fn,
                    )

                    predictions_decoded = decode_fn(predictions)
                    y = validation_batch["y"].cpu().numpy()
                    y_references = dataset.encoded_array_to_list_strings(y)
                    metrics = evaluate(predictions_decoded, y_references)
                    val_acc = np.mean(metrics["metric.accuracy"])
                    print(f"Val metrics: {val_acc}")

                    # save the model if we are at a saving step
                    if total_num_steps % CHECKPOINT_EVERY == 0:
                        save(
                            checkpoint_file=os.path.join(
                                checkpoints_dir,
                                "checkpoint_" + str(total_num_steps) + ".pt",
                            ),
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                        )
                    # update exponetial moving averages
                    ema_train_loss = (
                        (0.1 * train_loss.item() + 0.9 * ema_train_loss)
                        if ema_train_loss is not None
                        else train_loss.item()
                    )
                    ema_train_acc = (
                        (0.1 * train_acc + 0.9 * ema_train_acc)
                        if ema_train_acc is not None
                        else train_acc
                    )
                    ema_val_loss = (
                        (0.1 * val_loss.item() + 0.9 * ema_val_loss)
                        if ema_val_loss is not None
                        else val_loss.item()
                    )
                    ema_val_acc = (
                        (0.1 * val_acc + 0.9 * ema_val_acc)
                        if ema_val_acc is not None
                        else val_acc
                    )

                    log_row = pd.DataFrame(
                        [
                            {
                                "epoch": epoch,
                                "num_steps": total_num_steps,
                                "train_loss": train_loss.item(),
                                "ema_train_loss": ema_train_loss,
                                "val_loss": val_loss.item(),
                                "ema_val_loss": ema_val_loss,
                                "train_acc": train_acc,
                                "ema_train_acc": ema_train_acc,
                                "val_acc": val_acc,
                                "ema_val_acc": ema_val_acc,
                                "elapsed_time": elapsed_time,
                            }
                        ]
                    )
                    log_row.to_csv(
                        os.path.join(output_dir, "log.csv"),
                        mode="a",
                        header=False,
                        index=False,
                    )

    save(
        checkpoint_file=os.path.join(
            checkpoints_dir, "checkpoint_" + str(total_num_steps) + ".pt"
        ),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    print(f"Done, took {time.time() - start_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path where the data for the dataloaders is stored",
    )
    parser.add_argument(
        "--is-wick",
        action="store_true",
        help="is data-dir the path to R. Wick's preprocessed data",
    )
    parser.add_argument(
        "--is-synthetic",
        action="store_true",
        help="is data-dir the path to synthetic data generated by dataset_generator",
    )
    parser.add_argument(
        "--image-start",
        type=int,
        default=None,
        help="Start image for training on image data",
    )
    parser.add_argument(
        "--image-end",
        type=int,
        default=None,
        help="End image (exclusive) for training on image data",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path where the model is saved"
    )
    parser.add_argument(
        "--decoder",
        type=str,
        choices=["beam_search", "greedy"],
        required=True,
        help="Decoding method to use",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["ctc"],
        required=True,
        help="Loss function to use",
    )
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--starting-lr", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="checkpoint file to resume training",
    )
    parser.add_argument(
        "--finetune-checkpoint",
        type=str,
        help="checkpoint file to finetune from",
    )
    parser.add_argument(
        "--numpy-dir",
        type=str,
        help="Directory name where numpy files are located when --is-wick",
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        choices=["SACall", "SACallLoRA"],
        help="what model to use",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop training after a certain number of steps",
    )
    parser.add_argument(
        "--decoder-finetune",
        action="store_true",
        help="Only finetune the decoder",
        default=False,
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
        "-q", action="store_true", help="Enable LoRA on query projection", default=False
    )
    parser.add_argument(
        "-k", action="store_true", help="Enable LoRA on key projection", default=False
    )
    parser.add_argument(
        "-v", action="store_true", help="Enable LoRA on value projection", default=False
    )
    parser.add_argument(
        "-a",
        action="store_true",
        dest="attn_out",
        help="Enable LoRA on attention output",
        default=False,
    )
    parser.add_argument(
        "-f", action="store_true", help="Enable LoRA on FFN layers", default=False
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=GLOBAL_SEED,
        help="Random seed for training",
    )
    args = parser.parse_args()

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

    if args.is_wick and args.is_synthetic:
        raise ValueError("--is-wick and --is-synthetic are mutually exclusive")

    if (
        args.image_start is not None or args.image_end is not None
    ) and not args.is_synthetic:
        raise ValueError(
            "Can only use --image-start and --image-end with --is-synthetic"
        )

    if (
        (args.image_start is not None)
        and (args.image_end is None)
        and args.is_synthetic
    ):
        raise ValueError("--image-end must be specified")

    if args.decoder_finetune or args.bitfit or model_options["lora_options"].use_lora():
        if args.finetune_checkpoint is None:
            raise ValueError("To finetune, please provide --finetune-checkpoint")
        if args.checkpoint is not None:
            raise ValueError("To finetune, please provide --finetune-checkpoint")

    if args.finetune_checkpoint is not None and args.checkpoint is not None:
        raise ValueError(
            "To please provide one of --finetune-checkpoint or --checkpoint"
        )

    if args.model == "SACallLoRA":
        if (args.lora_rank is not None) and not (
            args.q or args.k or args.v or args.f or args.attn_out
        ):
            raise ValueError("Please specify lora options with -q -k -v -a -f")

    main(
        data_dir=args.path,
        is_wick=args.is_wick,
        is_synthetic=args.is_synthetic,
        output_dir=args.output_dir,
        decoder=args.decoder,
        loss=args.loss,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        starting_lr=args.starting_lr,
        warmup_steps=args.warmup_steps,
        checkpoint=args.checkpoint,
        finetune_checkpoint=args.finetune_checkpoint,
        numpy_dir=args.numpy_dir,
        is_profiling=args.profile,
        image_start=args.image_start,
        image_end=args.image_end,
        model_type=args.model,
        max_steps=args.max_steps,
        model_options=model_options,
        seed=args.seed,
    )
