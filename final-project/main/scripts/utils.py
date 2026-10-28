"""
util functions
"""

import os
import uuid
from collections import Counter
from contextlib import nullcontext
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from fvcore.nn import FlopCountAnalysis
from torch.profiler import DeviceType
from torch.utils.data import DataLoader
from tqdm import tqdm

from main.dataset_processing.base_fast5_dataset import BaseFast5Dataset
from main.dataset_processing.base_nanopore_dataset import BaseNanoporeDataset
from main.models.basecaller import Basecaller
from main.models.decoders import decode_ctc_beamsearch, decode_ctc_greedy
from main.models.evaluation import alignment_accuracy
from main.models.gradual_warmup_scheduler import GradualWarmupScheduler
from main.models.lora.model import SACallLoRA
from main.models.lora.modules import LoRAOptions
from main.models.losses import ctc_loss
from main.models.sacall.model import SACall
from main.scripts import fvcore_handlers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIPPING_VALUE = 2


def save(
    checkpoint_file: str,
    model: Basecaller,
    optimizer: torch.optim.Optimizer,
    scheduler: GradualWarmupScheduler,
):
    """Save the model state

    Copied from https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py
    """
    save_dict = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }

    torch.save(save_dict, checkpoint_file)


def load(
    checkpoint_file: str,
    model: Basecaller,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: GradualWarmupScheduler | None = None,
    initialize_lazy: bool = True,
):
    """Load a model state from a checkpoint file
    initialize_lazy (bool): to do a forward step before loading model,
        this solves the problem for layers that are initialized lazily

    Adapted from https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py
    """

    if initialize_lazy:
        dummy_batch = {"x": torch.randn([16, 1000], device=model.device)}
        predict_step(model=model, batch=dummy_batch)

    checkpoints = torch.load(checkpoint_file, weights_only=False)
    model.load_state_dict(checkpoints["model_state"])

    # do not load checkpoint for these if finetuning
    if optimizer is not None:
        optimizer.load_state_dict(checkpoints["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoints["scheduler_state"])


def create_dataset(
    data_dir: str,
    batch_size: int,
    is_wick: bool = False,
    is_synthetic: bool = False,
    numpy_dir: str = "",
    image_start: int | None = None,
    image_end: int | None = None,
) -> tuple[BaseNanoporeDataset, DataLoader, DataLoader]:
    """Initialise the dataset and train/val dataloaders

    Based on https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py
    """
    dataset = BaseNanoporeDataset(
        data_dir=data_dir,
        split=0.95,
        shuffle=True,
        seed=1,
        is_wick=is_wick,
        is_synthetic=is_synthetic,
        numpy_dir=numpy_dir,
        image_start=image_start,
        image_end=image_end,
    )

    dataloader_train = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=dataset.train_sampler,
        num_workers=1,
    )
    dataloader_validation = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=dataset.validation_sampler,
        num_workers=1,
    )

    return dataset, dataloader_train, dataloader_validation


def create_optimization(
    model: Basecaller,
    starting_lr: float,
    dataset: BaseNanoporeDataset,
    num_epochs: int,
    batch_size: int,
    warmup_steps: int,
) -> tuple[
    torch.optim.Optimizer,
    GradualWarmupScheduler,
]:
    """Initialise the optimizer, scheduler
    Based on https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=starting_lr)

    total_steps = (len(dataset.train_idxs) * num_epochs) / batch_size

    cosine_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_steps, eta_min=0.00001, last_epoch=-1
    )

    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=warmup_steps,
        after_scheduler=cosine_lr,
    )

    return (optimizer, scheduler)


def get_decoder(
    decoder: Literal["beam_search"] | Literal["greedy"],
) -> Callable[[torch.Tensor], list[str]]:
    if decoder == "beam_search":
        decode_fn = decode_ctc_beamsearch
    elif decoder == "greedy":
        decode_fn = decode_ctc_greedy

    return decode_fn


def get_loss(
    loss: Literal["ctc"],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if loss == "ctc":
        loss_fn = ctc_loss

    return loss_fn


def train_step(
    model: Basecaller,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: GradualWarmupScheduler,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    is_profiling: bool,
    count_flops: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, Any]]]:
    """Train a step with a batch of data
    batch (dict): dict with keys 'x' (batch, len)
                                    'y' (batch, len)
    Adapted from https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py
    """

    x = batch["x"].to(model.device)
    x = x.unsqueeze(1)  # add channels dimension
    y = batch["y"].to(model.device)

    # count flops
    if count_flops:
        model.eval()
        flops = FlopCountAnalysis(model, x)
        fvcore_handlers.register(flops)
    model.train()

    # forwards pass

    with (
        torch.profiler.record_function("forwards pass")
        if is_profiling
        else nullcontext()
    ):
        p = model(x, is_profiling)

    loss = loss_fn(y, p)

    # backwards pass
    optimizer.zero_grad()
    with (
        torch.profiler.record_function("backwards pass")
        if is_profiling
        else nullcontext()
    ):
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPPING_VALUE)
    optimizer.step()

    scheduler.step()
    return loss, p, (flops if count_flops else None)


def validation_step(
    model: Basecaller,
    batch: dict[str, torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
):
    """Predicts a single batch of data
    batch (dict): dict with keys 'x' (batch, len)
                                    'y' (batch, len)
    Copied from https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py
    """
    model.eval()
    with torch.no_grad():
        x = batch["x"].to(model.device)
        x = x.unsqueeze(1)  # add channels dimension
        y = batch["y"].to(model.device)

        p = model(x, is_profiling=False)  # forward through the network
        loss = loss_fn(y, p)

    return loss, p


def predict_step(model: Basecaller, batch: dict[str, torch.Tensor]):
    """
    Args:
        batch (dict) dict fill with tensor just for prediction
    Copied from https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py
    """
    model.eval()
    with torch.no_grad():
        x = batch["x"].to(model.device)
        x = x.unsqueeze(1)
        p = model.forward(x)
    return p


def evaluate(predictions: list[str], y_references: list[str]):
    """Evaluate the predictions by calculating the accuracy

    Args:
        predictions (list): list of predicted sequences as strings
        references (list): list of reference sequences as strings
    Copied from https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py
    """
    accs = []
    for i, sample in enumerate(y_references):
        accs.append(alignment_accuracy(sample, predictions[i]))

    return {"metric.accuracy": accs}


def stitch_by_stride(chunks, chunksize, overlap, length, stride, reverse=False):
    """
    Stitch chunks together with a given overlap

    This works by calculating what the overlap should be between two outputed
    chunks from the network based on the stride and overlap of the inital chunks.
    The overlap section is divided in half and the outer parts of the overlap
    are discarded and the chunks are concatenated. There is no alignment.

    Chunk1: AAAAAAAAAAAAAABBBBBCCCCC
    Chunk2:               DDDDDEEEEEFFFFFFFFFFFFFF
    Result: AAAAAAAAAAAAAABBBBBEEEEEFFFFFFFFFFFFFF

    Args:
        chunks (tensor): predictions with shape [samples, length, *]
        chunk_size (int): initial size of the chunks
        overlap (int): initial overlap of the chunks
        length (int): original length of the signal
        stride (int): stride of the model
        reverse (bool): if the chunks are in reverse order

    Copied from https://github.com/nanoporetech/bonito
    """

    if isinstance(chunks, np.ndarray):
        chunks = torch.from_numpy(chunks)

    if chunks.shape[0] == 1:
        return chunks.squeeze(0)

    semi_overlap = overlap // 2
    start, end = semi_overlap // stride, (chunksize - semi_overlap) // stride
    stub = (length - overlap) % (chunksize - overlap)
    first_chunk_end = (stub + semi_overlap) // stride if (stub > 0) else end

    if reverse:
        chunks = list(chunks)
        return torch.cat(
            [
                chunks[-1][:-start],
                *(x[-end:-start] for x in reversed(chunks[1:-1])),
                chunks[0][-first_chunk_end:],
            ]
        )
    else:
        return torch.cat(
            [
                chunks[0, :first_chunk_end],
                *chunks[1:-1, start:end],
                chunks[-1, start:],
            ]
        )


def split_bases_and_qscores(s: str) -> tuple[str, str]:
    """Given a string of potentially qscores & bases, split them"""
    # check if the quality scores are concatenated to the bases
    contains_qscores = False
    # first iterate through and check if we have any non-base characters
    for i in range(len(s)):
        if s[i] not in ["A", "C", "G", "T"]:
            contains_qscores = True
            break

    # if we have qscores then the first half must be the bases and the second half must be qscores
    if contains_qscores:
        bases = s[: len(s) // 2]
        qscores = s[len(s) // 2 :]
    else:
        # if we dont have any qscores craft fake ones
        bases = s
        qscores = "".join(["?"] * len(s))
    return bases, qscores


def decode_process(
    probs_stack: torch.Tensor,
    read_len: int,
    read_id,
    chunksize: int,
    overlap: int,
    stride: int,
    beam_size: int,
    beam_threshold: float,
    decoder: Literal["greedy"] | Literal["beam_search"],
    decoder_options: dict[str, Any],
) -> str:
    """Returns empty string if basecall failed else return the fastq string
    Adapted from https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py
    """
    probs_stack = stitch_by_stride(
        chunks=probs_stack,
        chunksize=chunksize,
        overlap=overlap,
        length=read_len,
        stride=stride,
        reverse=False,
    )
    probs_stack = probs_stack.unsqueeze(1)

    if decoder == "greedy":
        greedy = True
    else:
        greedy = False

    if greedy:
        seq = decode_ctc_greedy(p=probs_stack, qstring=True)
    else:
        seq = decode_ctc_beamsearch(
            p=probs_stack,
            beam_size=beam_size,
            beam_cut_threshold=beam_threshold,
            decoder_options=decoder_options,
        )

    if isinstance(seq[0], tuple):
        bases_seq = seq[0][0][: len(seq[0][1])]
        qscores = seq[0][0][len(seq[0][1]) :]
    else:
        bases_seq, qscores = split_bases_and_qscores(seq[0])

    id_line = "@" + str(read_id) + "\n"
    bases_line = bases_seq + "\n"
    separator_line = "+\n"
    qscores_line = qscores + "\n"
    fastq_string = id_line + bases_line + separator_line + qscores_line

    if qscores == "" or bases_seq == "":
        return ""

    return fastq_string


def basecall(
    model: Basecaller,
    dataset: BaseFast5Dataset,
    output_file: str,
    batch_size: int,
    decoder: Literal["greedy"] | Literal["beam_search"],
    decoder_options: dict[str, Any],
    verbose: bool = True,
    beam_size: int = 5,
    beam_threshold: float = 0,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Returns a list of all read_ids that failed to be read
    Adapted from https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py
    """
    # check if output file exists and raise error if overwrite is False
    if os.path.isfile(output_file):
        if overwrite:
            # clear file
            with open(output_file, "w"):
                pass
        else:
            raise FileExistsError(
                "Output file already exists, please use --overwrite if you wish to overwrite"
            )

    reads = []
    # iterate over the data
    for batch in tqdm(dataset, disable=not verbose):
        x = batch["x"].squeeze(0)
        l = x.shape[0]
        ss = torch.arange(0, l, batch_size)
        nn = ss + batch_size
        print(f"x before shape: {x.shape}")
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            p_list = list()
            p = predict_step(model, batch={"x": x})
            p_list.append(p)
        else:
            p_list = list()
            print(list(zip(ss, nn)))
            for s, n in zip(ss, nn):
                p = predict_step(model, batch={"x": x[s:n, :]})
                p_list.append(p)

        p = torch.hstack(p_list)

        ids = batch["id"]
        ids_arr = np.zeros((ids.shape[0],), dtype="U36")
        for i in range(ids.shape[0]):
            ids_arr[i] = str(uuid.UUID(fields=ids[i].tolist()))

        for read_id in np.unique(ids_arr):
            w = np.where(ids_arr == read_id)[0]
            read_stacks = p[:, w, :].permute(1, 0, 2)
            read_len = batch["len"][w[0]].item()

            fastq_string = decode_process(
                probs_stack=read_stacks,
                read_len=read_len,
                read_id=read_id,
                chunksize=dataset.window_size,
                overlap=dataset.window_overlap,
                stride=model.total_stride,
                beam_size=beam_size,
                beam_threshold=beam_threshold,
                decoder=decoder,
                decoder_options=decoder_options,
            )

            read_succeeded = 1 if fastq_string != "" else 0
            reads.append({"Read ID": read_id, "Read Succeeded": read_succeeded})
            if read_succeeded == 1:
                with open(output_file, "a") as f:
                    f.write(str(fastq_string))
                    f.flush()

    return pd.DataFrame(reads)


def trace_handler(profiler: torch.profiler.profile, model_tags: str, output_dir: str):
    """Updates the dataframe with profiler values"""
    # print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    # Filter to only ops inside the "forwards_pass" region
    df = pd.DataFrame(
        columns=[
            "Name",
            "Count",
            "Total CPU Time",  # microseconds
            "Total CUDA Time",  # microseconds
            "Total CPU Usage",  # Bytes
            "Total CUDA Usage",  # Bytes
        ]
    )
    for evt in profiler.events():
        if (
            (evt.name in ["forwards pass", "backwards pass"] + model_tags)
            and evt.is_user_annotation
            and evt.device_type == DeviceType.CPU
        ):
            row = {
                "Name": evt.name,
                "Count": evt.count,
                "Total CPU Time": evt.cpu_time_total,
                "Total CUDA Time": evt.device_time_total,
                "Total CPU Usage": evt.cpu_memory_usage,
                "Total CUDA Usage": evt.device_memory_usage,
            }
            df.loc[len(df)] = row
    df = df.groupby("Name", as_index=False).sum()
    df.to_csv(
        os.path.join(output_dir, "profile.csv"),
        mode="a",
        header=False,
        index=False,
    )


def convert_counter_to_dict(obj):
    if isinstance(obj, Counter):
        return dict(obj)
    elif isinstance(obj, dict):
        return {key: convert_counter_to_dict(value) for key, value in obj.items()}
    else:
        return obj


def get_model(
    model_type: Literal["SACall"] | Literal["SACallLoRA"],
    training: bool,
    model_options: dict[str, Any],
    starting_lr: float | None = None,
    dataset: BaseNanoporeDataset | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    warmup_steps: int | None = None,
    checkpoint: str | None = None,
    finetune_checkpoint: str | None = None,
) -> tuple[
    Basecaller,
    torch.optim.Optimizer | None,
    GradualWarmupScheduler | None,
]:
    """Returns the model to use"""
    # intialise model
    model = None
    if model_type == "SACall":
        model = SACall(
            d_model=256,
            d_ff=1024,
            n_head=8,
            num_encoder_layers=6,
            device=device,
        )
    elif model_type == "SACallLoRA":
        model = SACallLoRA(
            d_model=256,
            d_ff=1024,
            n_head=8,
            num_encoder_layers=6,
            device=device,
        )
    model = model.to(device)
    optimizer = None
    scheduler = None
    lora_options: LoRAOptions = model_options["lora_options"]
    use_lora = lora_options.use_lora()
    if training:
        if finetune_checkpoint is not None:
            # load pre-trained weights only (fresh optimizer/scheduler below)
            load(
                checkpoint_file=finetune_checkpoint,
                model=model,
                initialize_lazy=True,
            )

            if model_options["decoder_finetune"] or use_lora or model_options["bitfit"]:
                # freeze all parameters, then unfreeze the selected ones
                for param in model.parameters():
                    param.requires_grad = False

                if model_options["decoder_finetune"]:
                    model.decoder_finetune()
                if model_options["bitfit"]:
                    model.bitfit()
                if use_lora:
                    if not isinstance(model, SACallLoRA):
                        raise ValueError("Must use model SACallLoRA to use LoRA")
                    model.finetune(lora_options)
            # else: whole-model fine-tuning, all parameters stay trainable

        # Initialise optimizer, scheduler,
        print("Creating optimization")
        optimizer, scheduler = create_optimization(
            model=model,
            starting_lr=starting_lr,
            dataset=dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
        )

        # If checkpoint provided then load it
        if checkpoint is not None:
            load(
                checkpoint_file=checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                initialize_lazy=True,
            )
    else:
        if model_type == "SACallLoRA":
            if lora_options.rank is not None:
                model.finetune(lora_options)
            if model_options["bitfit"]:
                model.bitfit(train=False)
        model = model.to(device)
        load(
            checkpoint_file=checkpoint,
            model=model,
            initialize_lazy=True,
        )

    model.to(device)

    return model, optimizer, scheduler
