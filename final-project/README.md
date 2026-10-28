## Getting started
```
python -m venv venv
source venv/activate/bin
pip install -r requirements
```

### Install torch
Check [pytorch website](https://pytorch.org/get-started/locally/) for installation (depends on cuda version)

### Install fast-ctc-decode fork
```
git clone https://github.com/advait-menon/fast-ctc-decode.git
cd fast-ctc-decode
make develop
```
### Install bio tools
Tombo and Flye installation:
```
# first exit any venv
source deactivate
conda create -n tools python=2.7
conda activate tools
conda install -c bioconda ont-tombo
conda install -c bioconda flye
```
Install samtools/bcftools/htslib from https://www.htslib.org/download/ and add them to PATH. 

Minimap2 installation:
```
curl -L https://github.com/lh3/minimap2/releases/download/v2.28/minimap2-2.28_x64-linux.tar.bz2 | tar -jxvf -
```
Then extract the minimap2, paftools.js, and k8 binaries to same location as the samtools/bcftools/htslib binaries, and add them to PATH.

Medaka installation:
```
conda create -n medaka python=3.9
conda activate medaka
pip install medaka
```

Download the [slow5tools](https://github.com/hasindu2008/slow5tools) and [Squigulator](https://github.com/hasindu2008/squigulator) binaries and place them in `tools/`

## Dataset Setup:
### Bacterial Dataset

Download [Wick et al. dataset](https://bridges.monash.edu/projects/Performance_of_neural_network_basecalling_tools_for_Oxford_Nanopore_sequencing/59618) with the script (copied from https://github.com/marcpaga/nanopore_benchmark). CAUTION: this is around 400GB when fully downloaded:
```
cd main/scripts
./download_script.sh <data dir>
```
Generate fastq files for each genome:
```
python main/scripts/create_fastqs.py --path <data dir> --verbose
```
Annotate and preprocess with tombo:
```
python main/scripts/tombo_processing.py \
--path <data dir> --verbose --processes 4
```
Generate numpy arrays:
```
python -m main.scripts.data_prepare_numpy \
--path <data dir> \
--total-files 200 --window-size 2000 --window-slide 0 \
--n-cores 8 --verbose --is-wick \
--out-dir nn_inputs
```
### Simulated datasets
#### Evaluation Dataset
Download Goldman et al. [files](https://www.nature.com/articles/nature11875#Sec2) and extract the 5 files into `./goldman_data`

Generate the simulated data:
```
python -m main.scripts.dataset_generator --input-dir ./goldman_data \
--output-dir <data dir> \
--path-to-code ./codes/code.cd --threads 10 --encoding-method goldman 
```
#### Training Dataset
Download the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) python version and extract it.

Save the images:
```
python ./main/scripts/get_cifar_images.py \
--cifar-dir <path to cifar-10-batches-py> \
--output-dir <image dir> --batches 1
```

Simulate the FAST5 squiggle data:
```
python -u -m main.scripts.dataset_generator \
--input-dir <image dir> \
--output-dir <data dir> \
--path-to-code ./codes/code.cd \
--single-fast5 \
--encoding-method goldman \
--threads 2
```

Pre-process and re-squiggle with Tombo:
```
conda activate tombo
python -u main/scripts/tombo_synthetic.py \
--path <data dir> \
--processes 50 --verbose
```

## Training
### Training on bacterial dataset
Train an SACAll benchmark model on bacterial data:
```
python -m main.scripts.train_model --path <data dir> \
--is-wick --output-dir <model dir> \
--decoder "beam_search" --loss "ctc" \
--num-epochs 4 --numpy-dir nn_inputs \
--model SACallLoRA
```
### Fine-tuning on simulated CIFAR-10 dataset
Whole-model fine-tuning:
```
python -m main.scripts.train_model --path <data dir> \
--is-synthetic --output-dir <model dir> \
--decoder "beam_search" --loss "ctc" \
--num-epochs 100000 --numpy-dir nn_inputs \
--model SACallLoRA --max-steps 45000 \
--image-start 0 --image-end 5000 \
--checkpoint <benchmark model dir>/checkpoints/<checkpoint>.pt
```

Decoder fine-tuning:
```
python -m main.scripts.train_model --path <data dir> \
--is-synthetic --output-dir <model dir> \
--decoder "beam_search" --loss "ctc" \
--num-epochs 100000 --numpy-dir nn_inputs \
--model SACallLoRA --max-steps 45000 \
--image-start 0 --image-end 5000 \
--finetune-checkpoint <benchmark model dir>/checkpoints/<checkpoint>.pt \
--decoder-finetune
```

LoRA fine-tuning:
```
python -m main.scripts.train_model --path <data dir> \
--is-synthetic --output-dir <model dir> \
--decoder "beam_search" --loss "ctc" \
--num-epochs 100000 --numpy-dir nn_inputs \
--model SACallLoRA --max-steps 45000 \
--image-start 0 --image-end 5000 \
--finetune-checkpoint <benchmark model dir>/checkpoints/<checkpoint>.pt \
--lora-rank 4 \
-qk
```
## Basecalling
Basecalling script:
```
python -u -m main.scripts.basecall \
--path <data dir> \
--checkpoint <checkpoint path> \
--output <basecall dir> \
--chunk-size 2000 \
--window-overlap 0 \
--batch-size 64 \
--beam-size 3 \
--beam-threshold 0.0 \
--verbose \
--decoder beam_search \
--model SACallLoRA
```
If evaluating on Wick et al. or simulated evaluation dataset use:
```
--is-wick
--is-synthetic
```
If basecalling a LoRA model use:
```
--lora-rank <rank> -qk
```
### Beam search modifications
Use the following options to enable the beam search modifications:
```
--homopolymer-penalty <penalty value> # penalty to extension if homopolymer length exceeds max_repeats
--max-repeats <length> # max homopolymer length allowed, use in conjunction with --homopolmyer-penalty to penalise, or alone for discarding extensions
--desired-length <length> # what is the desired length of the DNA sequences (e.g. 117nt), use with --length-lambda and --discard-expected-length
--length-lambda <lambda> # penalty coefficient for penalising extensions by normalised length
--discard-by-expected-length <theta> # threshold for discarding extensions if they differ too much from the expected length at that time step
--kl-lambda <lambda> # penalty coefficient for KL-divergence penalty
--discard-kl <theta> # KL threshold for discarding extensions that diverge too far from the uniform distribution
--entropy-ratio <ratio> # the proportion of the max entropy to use for the entropy threshold e.g. 0.8
```

## Evaluation
Most metrics for basecalling Wick et al. dataset or the simulated evaluation dataset can be computed with:

```
python -u -m main.scripts.evaluate_model \
--reference <data dir> \
--predictions <basecall dir> \
--verbose \
--is-wick or --is-synthetic # for Wick et al. or simulated
```

For Wick et al. dataset we can compute the consensus accuracy:

```
# Run Flye to generate assemblies
conda activate tombo
python -u -m main.scripts.generate_assemblies \
--path <basecall dir> \
--threads 20 \
--verbose \
--is-wick \
--genome-path <data dir> \
--asm-coverage 40
conda deactivate

# Run Medaka to create consensus sequence
conda activate medaka
python -u -m main.scripts.run_medaka \
--path <basecall dir> \
--threads 30 \
--verbose \
--is-wick

conda deactivate
conda deactivate

# compute consensus accuracies
python -u -m main.scripts.consensus_accuracy \
--genome-path <data dir> \
--assembly-path <basecall dir> \
--chunk-size 10000 \
--verbose \
--is-wick
```

And for the evaluation dataset, we can compute the payload accuracies:

```
python -m main.scripts.payload_accuracy_synthetic \
--input-dir <basecall dir> \
--goldman-dir <data dir>
```

## Scripts
The scripts in `main/scripts/` also allow for training, basecalling and evaluating on custom datasets. For more information, refer to the source code of each script, or use the `--help` option.

## Reproducing the paper tables (multi-seed)

The paper's tables report mean ± SD over **11 simulated evaluation (test) datasets** (seeds `1–10` and `42`). Table 1 also uses **3 fine-tuning (training) seeds** (`1`, `2`, `42`). Five shell scripts in `final-project/` run the experiments. The notebook `notebooks/regenerate_tables_multiseed.ipynb` then turns the results into LaTeX tables and figures, and saves them in the `Reports/` folder.

### Requirements

- **Python 3.10+**: the code uses `str | None` type hints.
- **A CUDA GPU**: strongly recommended. Basecalling falls back to CPU, but it is impractically slow at this scale. The scripts run several seeds in parallel on the same GPU.
- **Bash 4+**: the scripts use `set -euo pipefail`, `export -f` and `xargs -P`. On macOS, the default `/bin/bash` 3.2 is too old, so install a newer bash (e.g. `brew install bash`). Linux or WSL is recommended.
- Everything in [Getting started](#getting-started), i.e.:
  - `pip install -r requirements.txt`
  - PyTorch matching your CUDA version
  - the `fast-ctc-decode` fork, which provides the constraint-aware beam search. Building it needs the Rust toolchain (https://rustup.rs) and `maturin`. A local copy is in `../cloned_repos/fast-ctc-decode`, so you can run `cd ../cloned_repos/fast-ctc-decode && make develop`.
  - the `squigulator` and `slow5tools` binaries in `tools/` (these are only needed if you generate datasets)
- **Jupyter**, to run the notebook. It is not in `requirements.txt`:
  ```
  pip install jupyter
  ```
  The notebook itself only needs `numpy`, `pandas` and `matplotlib`, which are all in `requirements.txt`. It does not need a GPU, PyTorch or the training data.
- **The pre-trained benchmark model**, `../checkpoint_570209.pt`. This is the SACall model trained on the Wick et al. bacterial dataset. It is the default `CKPT` / `BASE_CKPT` in every script.
- **The Goldman et al. source files** in `./goldman_data` (see [Evaluation Dataset](#evaluation-dataset)).

### Expected folder layout

Run every script **from inside `final-project/`**. All paths are relative to that folder (`../data`, `../experiments`, `./codes`, `./tools`).

```
<repo>/
├── checkpoint_570209.pt                     # pre-trained benchmark model
├── data/
│   └── goldman_synthetic_data_seed_{1..10,42}/   # 11 simulated evaluation datasets
├── experiments/
│   ├── finetune_multiseed/seed_{1,2,42}/<model>/  # fine-tuned models (+ log.csv, profile.csv)
│   ├── table61_eval_3train_10test/          # Table 1  (table61_repeats_parallel.sh)
│   ├── table71_repeats/                     # Table 4  (run_table71_11seeds.sh)
│   ├── table73_74_repeats/                  # Tables 5, 6 (table73_74_repeats_parallel.sh)
│   ├── table75_76_repeats/                  # Tables 7, 8 (table75_76_repeats_parallel.sh)
│   ├── table77_repeats/                     # Table 9  (table77_repeats_parallel.sh)
│   ├── s4_combined/train_{1,2,42}/test_1/<model>[_combined]/   # S4 Table
│   └── finetune/                            # data-regime runs (S2 Appendix)
├── Reports/
│   └── Final Report/                        # notebook output (see below)
└── final-project/                           # <- run scripts from here
    ├── *.sh
    └── notebooks/regenerate_tables_multiseed.ipynb
```

Each evaluation folder contains one sub-folder per encoded file (`EBI`, `MLK_excerpt_VBR_45-85`, `View_huff3.cd`, `watsoncrick`, `wssnt10`). Each of those holds `basecalls.fastq`, `basecall_stats.csv`, `evaluation/alignment_stats.csv`, `evaluation/evaluation.csv` and `payload_accuracy.txt`. The notebook reads these files.

### Step 1: Run the experiment scripts

Each script basecalls with `main.scripts.basecall`, computes read metrics with `main.scripts.evaluate_model`, and computes payload accuracy with `main.scripts.payload_accuracy_synthetic`. It does this for every seed and every beam-search setting. Seeds run in parallel. The per-seed logs are written to `<EXP_ROOT>/logs/seed_<N>.log`. A failing seed is reported as `[FAILED]` and makes the script exit with a non-zero status.

| Script | Paper table(s) | What it runs | Evaluations | Output (`EXP_ROOT` / `OUT_ROOT`) |
|---|---|---|---|---|
| `table77_repeats_parallel.sh` | Table 9 | **Generates** the 11 datasets `../data/goldman_synthetic_data_seed_<N>`, then runs the default decoder plus the combined beam search (`--max-repeats 1`, length λ ∈ {5,10,15} × KL λ ∈ {1,5,10}) | 110 | `../experiments/table77_repeats` |
| `run_table71_11seeds.sh` | Table 4 | Homopolymer constraint: penalty ∈ {0.2, 0.5} × max repeats ∈ {1, 2}, plus penalty 1 / max repeats 1 | 55 | `../experiments/table71_repeats` |
| `table73_74_repeats_parallel.sh` | Tables 5, 6 | Length penalty λ ∈ {1,5,10,15,20,25} and length discard θ ∈ {0.05,0.1,0.2,0.5} (desired length 117 nt) | 121 | `../experiments/table73_74_repeats` |
| `table75_76_repeats_parallel.sh` | Tables 7, 8 | KL penalty λ ∈ {1,5,10,15,20} and KL discard θ ∈ {0.01,0.05,0.1} (entropy ratio 0.8) | 99 | `../experiments/table75_76_repeats` |
| `table61_repeats_parallel.sh` | Table 1 | Baseline plus 7 fine-tuned models (whole-model, decoder, LoRA ranks 1/2/4/8/16) × 3 training seeds × 11 test seeds | 242 | `../experiments/table61_eval_3train_10test/run_<timestamp>_XXXXXX` |

**Run order.** `table77_repeats_parallel.sh` is the only script that creates the 11 evaluation datasets, so run it first. The other four scripts reuse those datasets and stop if any of them is missing.

> ⚠️ `table77_repeats_parallel.sh` deletes and regenerates `../data/goldman_synthetic_data_seed_<N>` and `../experiments/table77_repeats/seed_<N>` for every seed (`rm -rf`). The scripts for Tables 4–8 use `--overwrite`, so they overwrite their own earlier results. `table61_repeats_parallel.sh` never overwrites anything: each run writes to a new timestamped `run_*` folder.

Example:
```
cd final-project
chmod +x *.sh
./table77_repeats_parallel.sh        # generates datasets + Table 9
./run_table71_11seeds.sh             # Table 4
./table73_74_repeats_parallel.sh     # Tables 5, 6
./table75_76_repeats_parallel.sh     # Tables 7, 8
./table61_repeats_parallel.sh        # Table 1 (needs fine-tuned checkpoints, see below)
```

All settings can be overridden with environment variables. This is useful for a smaller GPU or for non-default paths:

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `CKPT` | `../checkpoint_570209.pt` | Tables 4–9 | Benchmark model checkpoint |
| `BASE_CKPT` | `../checkpoint_570209.pt` | Table 1 | Baseline checkpoint |
| `DATA_ROOT` | `../data` | all | Folder containing `goldman_synthetic_data_seed_<N>` |
| `INPUT_DIR` | `./goldman_data` | Tables 4, 7–9 | Goldman et al. source files |
| `CODE` | `./codes/code.cd` | Tables 4, 7–9 | Prefix code used for encoding |
| `EXP_ROOT` / `OUT_ROOT` | see table above | all | Where results are written |
| `TRAIN_ROOT` | `../experiments/finetune_multiseed` | Table 1 | Fine-tuned checkpoints |
| `PARALLEL_JOBS` | `2` (Tables 1, 4), `4` (others) | all | Seeds evaluated at the same time on the GPU |
| `BATCH_SIZE` | `64` | all | Basecalling batch size |
| `THREADS` | `10` | Tables 4, 7–9 | Threads for dataset generation |

e.g. `PARALLEL_JOBS=1 BATCH_SIZE=32 ./table73_74_repeats_parallel.sh`

**Notes before running:**
- **Dataset seeds.** `table77_repeats_parallel.sh` calls `main.scripts.dataset_generator ... --seed <N>`, and that seed is passed to Squigulator. Without `--seed`, it defaults to `SQUIGULATOR_SEED = 42` from `main/dataset_processing/seeds.py`. Each dataset writes its temporary BLOW5 files to its own `<output-dir>/temp_blow5`, so seeds can be generated in parallel safely.
- **Table 1 checkpoints.** `table61_repeats_parallel.sh` needs 21 fine-tuned checkpoints at `../experiments/finetune_multiseed/seed_{1,2,42}/<model>/checkpoints/checkpoint_45000.pt`. Here `<model>` is one of `5000_bacterial_benchmark_finetuned_45000`, `5000_images_model_decoder_finetune_45000` or `5000_images_model_finetune_rank_{1,2,4,8,16}_45000`. Train them with the commands in [Fine-tuning on simulated CIFAR-10 dataset](#fine-tuning-on-simulated-cifar-10-dataset), once per training seed, with `--output-dir ../experiments/finetune_multiseed/seed_<seed>/<model>`. The script checks that all 21 checkpoints and all 11 datasets exist before it starts.
- **Table 1 output folder.** Table 1 results are written to a new folder, `../experiments/table61_eval_3train_10test/run_<timestamp>_XXXXXX/`. The notebook reads `experiments/table61_eval_3train_10test/{baseline,train_*}` directly. So before running the notebook, either move the contents of the chosen `run_*` folder up one level, or set `T1_ROOT` in the notebook to that `run_*` folder.

### Step 2: Regenerate the tables with the notebook

`notebooks/regenerate_tables_multiseed.ipynb` reads the result files above. It does not train or basecall anything.

1. Open the notebook (`jupyter notebook notebooks/regenerate_tables_multiseed.ipynb`).
2. In the first code cell, set `PROJECT_ROOT` to the absolute path of your `final-project/` folder. The notebook works out `experiments/` and `Reports/` from it, as siblings of `final-project/`.
3. Restart the kernel and run all cells in order.

The notebook:
- loads every result once. It **stops with an error** if any required file or seed is missing, instead of silently using fewer seeds.
- computes mean ± SD over the 11 test seeds (and over the 3 training seeds for Table 1).
- computes paired statistics against the baseline for every configuration: mean difference, 95% t-interval, exact sign-flip test and Holm-adjusted p-values.
- builds the S3 training-curve figures (mean ± SD over training seeds) from `finetune_multiseed/seed_*/<model>/log.csv`.
- builds the S4 Table from `experiments/s4_combined/train_{1,2,42}/test_1/<model>` and `<model>_combined`.

Table 3 (training cost) is read from the existing `experiments/finetune_multiseed/seed_42/<model>/profile.csv` files.

### Step 3: Outputs in `Reports/`

Everything is saved in `<repo>/Reports/Final Report/`:

```
Reports/Final Report/
├── tables_fixed/
│   ├── Table_01.tex … Table_09.tex   # main-text tables (LaTeX fragments to paste into Overleaf)
│   ├── S1_Table.tex                  # supporting tables (standalone LaTeX documents)
│   ├── S4_Table.tex
│   └── S5_Table.tex                  # paired statistical comparisons vs. baseline
├── S5_Table.tex                      # S5 copy / CSV kept from an earlier run
├── S5_statistical_comparisons.csv    #   (not written by the current notebook)
└── images/
    ├── whole_model_finetune_loss.svg # S3 Fig: training/validation loss
    └── whole_model_finetune_acc.svg  # S3 Fig: training/validation read accuracy
```

| File | Paper table |
|---|---|
| `Table_01.tex` | Fine-tuning methods (baseline + 7 models, 3 training seeds × 11 test seeds) |
| `Table_02.tex` | Trainable parameter counts |
| `Table_03.tex` | Training cost (from `profile.csv`) |
| `Table_04.tex` | Homopolymer constraint |
| `Table_05.tex` / `Table_06.tex` | Length penalty / length discard |
| `Table_07.tex` / `Table_08.tex` | KL penalty / KL discard |
| `Table_09.tex` | Combined constraints |
| `S1_Table.tex`, `S4_Table.tex`, `S5_Table.tex` | Supporting information tables |

The main tables are fragments to paste into the manuscript. They need the LaTeX packages `booktabs`, `graphicx`, `amsmath`, `changepage` and `xcolor` (with the `table` option). The `S*_Table.tex` files are complete documents that compile on their own.

`notebooks/regenerate_tables.ipynb` is the earlier single-seed version of the table notebook. `notebooks/wick_dataset_eval.ipynb` plots the training curves of the bacterial benchmark model (from `../experiments/bacterial_benchmark/log.csv`) into `notebooks/figures/`.

## Acknowledgements

This repository includes code adapted from the following projects:

- [basecalling_architectures](https://github.com/marcpaga/basecalling_architectures) by Marc Pagès‑Gallego – used in the underlying training and basecalling scripts.
- [nanopore_benchmark](https://github.com/marcpaga/nanopore_benchmark) by Marc Pagès‑Gallego – the download script for Wick et al. dataset is copied from here.
- [SACall](https://github.com/huangnengCSU/SACall-basecaller) by Neng Huang – used as a reference for the SACall and TransformerEncoderLayer implementations.
- [Bonito](https://github.com/nanoporetech/bonito) by Oxford Nanopore Technologies – used in the underlying training and basecalling scripts.
- [Gradual Warmup Scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr) by Ildoo Kim - used in the underlying training script.

All credit goes to the original authors.
