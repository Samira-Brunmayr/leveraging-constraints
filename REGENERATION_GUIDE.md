# Regeneration Guide — Basecalling for DNA Storage

> **Note:** This guide covers the original **single-seed** workflow. The paper's tables
> report mean ± SD over 11 evaluation seeds (and 3 fine-tuning seeds for Table 1). That
> **multi-seed** workflow (the `.sh` scripts in `final-project/` and
> `notebooks/regenerate_tables_multiseed.ipynb`) is documented in `final-project/README.md`,
> section "Reproducing the paper tables (multi-seed)".

This is a complete, ordered runbook for regenerating every result needed to reproduce
Tables 1–9 of the paper/thesis from scratch, using only the code in this repo plus
publicly downloadable source data. It was reconstructed by cross-referencing
`README.md`, the recovered `.sh` scripts in the project root, the Python source under
`main/scripts/`, and `notebooks/regenerate_tables.ipynb` (the actual table generator).

**This guide sits at the top level of the shared package**, alongside `final-project/`,
the root `.sh` scripts, `checkpoint_570209.pt`, and `cloned_repos/fast-ctc-decode/` —
`<repo>` below refers to this top-level folder (whatever you name it after extracting).
**Assumed working directory for all commands below: `<repo>/final-project/`** — every
script uses `../data`, `../experiments`, `./codes`, `./tools` relative paths that only
resolve correctly from inside that folder.

**Two repos are involved.** `final-project/` (published on GitHub as
[`advait-menon/leveraging-constraints`](https://github.com/advait-menon/leveraging-constraints))
is the main basecalling/training/evaluation codebase. The constraint-aware beam search
itself (homopolymer/length/KL-divergence modifications) lives in a *separate* fork of
Oxford Nanopore's `fast-ctc-decode` library, published as
[`advait-menon/fast-ctc-decode`](https://github.com/advait-menon/fast-ctc-decode) — a
local copy is already checked out at `<repo>/cloned_repos/fast-ctc-decode/`, so no
`git clone` is needed if you're working from this folder tree directly.

**One piece of good news up front**: `<repo>/checkpoint_570209.pt` (the pre-trained
benchmark model) survived the deletion. That alone unblocks Tables 4–9 and the two
baseline rows in Table 1, once the evaluation dataset is regenerated — no retraining
required for those.

---

## Phase 0 — Environment setup

**Not stated anywhere in the README**: the codebase uses `str | None`-style type hints
throughout (`train_model.py`, `utils.py`, etc.), which requires **Python 3.10+**. Using
an older `python3` for the venv below will fail with syntax errors partway through, not
at setup time.

A **CUDA-capable GPU** is also effectively required — the code falls back to CPU
(`torch.device("cuda" if torch.cuda.is_available() else "cpu")`) but training/basecalling
at these scales is impractical without one (the paper used a Tesla A30).

From `README.md`:

```bash
python3 -m venv venv   # must be Python 3.10+
source venv/bin/activate
pip install -r requirements.txt
```

- Install PyTorch per the [PyTorch site](https://pytorch.org/get-started/locally/) (match your CUDA version) — it's not in `requirements.txt`, install it separately in the same venv.
- **Jupyter is also not in `requirements.txt`**, but Phase 7 needs it to run `regenerate_tables.ipynb`:
  ```bash
  pip install jupyter
  ```
- `download_script.sh` (Phase 1a) shells out to `wget` — make sure it's on PATH (usually preinstalled on Linux, but not guaranteed on a minimal/container image).
- Install the custom beam-search decoder fork (this is *not* upstream fast-ctc-decode — it has the constraint-aware modifications baked in). It's a Rust project built with `maturin` (already in `requirements.txt`), so it needs the **Rust toolchain** installed first — this isn't mentioned in the main README, but `make develop` will fail without it:
  ```bash
  # install Rust if you don't have it (https://rustup.rs)
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"   # rustup won't be on PATH in this shell session otherwise

  # with the venv from above still active:
  cd ../cloned_repos/fast-ctc-decode   # local copy, no git clone needed
  make develop                          # = maturin develop -F python --release
  cd ../../final-project
  ```
  If you're setting this up from a fresh `git clone` instead of this folder tree, replace
  the `cd` line with `git clone https://github.com/advait-menon/fast-ctc-decode.git && cd fast-ctc-decode`.
- Bio tools (separate conda env, Python 2.7 required by Tombo — assumes conda/miniconda is already installed):
  ```bash
  conda create -n tools python=2.7
  conda activate tools
  conda install -c bioconda ont-tombo
  conda install -c bioconda flye
  ```
- `samtools`/`bcftools`/`htslib` — the README just says "download and add to PATH" without commands; these are a source build (standard `./configure && make && make install`, needs `zlib`/`libbz2`/`liblzma`/`libcurl` dev headers on the system):
  ```bash
  curl -L https://github.com/samtools/htslib/releases/download/1.21/htslib-1.21.tar.bz2 | tar -jxvf -
  cd htslib-1.21 && ./configure && make && sudo make install && cd ..
  # repeat the same pattern for samtools and bcftools (same release train, same site)
  ```
  No `sudo` on a shared/university machine? Use `./configure --prefix="$HOME/.local"`
  instead, skip `sudo` on the install step, and make sure `$HOME/.local/bin` is on PATH.
- Minimap2 (**not actually needed for Tables 1–9** — checked: `evaluate_model.py` and
  `metrics.py` only `import mappy`, the Python package already in `requirements.txt`,
  which bundles its own compiled minimap2 library; nothing in `main/` shells out to a
  standalone `minimap2`/`paftools.js`/`k8` binary. The README lists this as a required
  setup step, but it's only relevant if you're also redoing the bacterial
  consensus-accuracy comparison via Flye, which isn't part of the paper's 9 tables. Skip
  it unless you hit an actual error that needs it):
  ```bash
  curl -L https://github.com/lh3/minimap2/releases/download/v2.28/minimap2-2.28_x64-linux.tar.bz2 | tar -jxvf -
  ```
  (extract `minimap2`, `paftools.js`, `k8` next to the htslib binaries, on PATH)
- Medaka (separate env, only needed for the bacterial consensus-accuracy table, not Tables 1–9):
  ```bash
  conda create -n medaka python=3.9
  conda activate medaka
  pip install medaka
  ```
- `slow5tools` and `Squigulator` binaries into `tools/` — a `slow5tools-v1.3.0-release.tar.gz` and a `tools/squigulator`, `tools/slow5tools` already exist in the recovered tree; verify they're executable (`chmod +x tools/squigulator tools/slow5tools`).

---

## Phase 1 — Source data acquisition

### 1a. Bacterial pre-training/benchmark data (only needed if you also want to redo Table 3's "default"/benchmark profiling row, or reproduce the benchmark from scratch — **not** needed for Tables 1, 4–9 since `checkpoint_570209.pt` already exists)

```bash
cd main/scripts
./download_script.sh <data dir>   # CAUTION: ~400GB
cd ../..
```
Sourced from Wick et al.'s [Bridges dataset](https://bridges.monash.edu/projects/Performance_of_neural_network_basecalling_tools_for_Oxford_Nanopore_sequencing/59618) — direct file links are already saved in `main/scripts/links_wick_*.txt` (the download script reads these).

### 1b. Goldman evaluation files (needed for **all** of Tables 1, 4–9)

Five files, from the supplementary data of Goldman et al. 2013 (Nature): https://www.nature.com/articles/nature11875#Sec2

| File | Already recovered? |
|---|---|
| `wssnt10.txt` | No — download |
| `watsoncrick.pdf` | No — download |
| `EBI.jp2` | No — download |
| `MLK_excerpt_VBR_45-85.mp3` | No — download |
| `View_huff3.cd.new` | **Yes** — already at `codes/View_huff3.cd.new` |

Create `./goldman_data/` and put all 5 files in it (copy `codes/View_huff3.cd.new` in too — the pipeline expects all 5 together in this one folder).

### 1c. CIFAR-10 (needed only for Table 1's LoRA/Whole-Model/Decoder rows and Table 3 — i.e. if you're retraining the fine-tuned checkpoints; **not** needed for Tables 4–9)

Download the Python version from https://www.cs.toronto.edu/~kriz/cifar.html and extract it.

---

## Phase 2 — Data preprocessing

### 2a. Goldman synthetic evaluation dataset (needed for Tables 1, 4–9)

```bash
python -m main.scripts.dataset_generator --input-dir ./goldman_data \
  --output-dir ../data/goldman_synthetic_data \
  --path-to-code ./codes/code.cd --threads 10 --encoding-method goldman

conda activate tools   # tombo env
python -u main/scripts/tombo_synthetic.py \
  --path ../data/goldman_synthetic_data \
  --processes 50 --verbose
conda deactivate
```
Squigulator params are **hardcoded** in `dataset_generator.py` (`-x dna-r9-min --full-contigs --seed 42 --dwell-mean 13`) — matches the thesis exactly, no flags to remember.

> **Update:** `dataset_generator.py` now accepts a `--seed` option, which is passed to
> Squigulator (default `42`). The multi-seed scripts use it to generate one dataset per
> seed; see `final-project/README.md`.

**Undocumented step required for Phase 6 to work at all**: `dataset_generator.py`'s
`file_to_fastq()` writes the reference genome to `genomes/<name>.fna`, but
`payload_accuracy_synthetic.py` (Phase 6) looks for the reference at
`fastq/<name>.fasta` — different directory *and* different extension. Nothing in the
codebase bridges this; it's not in the README or any script. Payload accuracy is a
required column in every table, so without this, Phase 6 fails with
`FileNotFoundError` on every sample. `.fna` and `.fasta` are the same format (Biopython's
`SeqIO.parse(..., "fasta")` doesn't care about the extension), so a plain copy fixes it:
```bash
for f in ../data/goldman_synthetic_data/genomes/*.fna; do
  name=$(basename "$f" .fna)
  cp "$f" "../data/goldman_synthetic_data/fastq/${name}.fasta"
done
```
Only needed for `goldman_synthetic_data` (the evaluation set) — `payload_accuracy_synthetic.py`
is always run against this directory as `--goldman-dir`, regardless of which model's
predictions are being scored, so `cifar_synthetic_data` never needs this.

### 2b. CIFAR-10 → synthetic fine-tuning training dataset (only if retraining for Table 1/3)

```bash
python ./main/scripts/get_cifar_images.py \
  --cifar-dir <path to cifar-10-batches-py> \
  --output-dir <image dir> --batches 1

python -u -m main.scripts.dataset_generator \
  --input-dir <image dir> \
  --output-dir ../data/cifar_synthetic_data \
  --path-to-code ./codes/code.cd \
  --single-fast5 \
  --encoding-method goldman \
  --threads 2

conda activate tools
python -u main/scripts/tombo_synthetic.py \
  --path ../data/cifar_synthetic_data \
  --processes 50 --verbose
conda deactivate
```

**Missing step the README doesn't mention**: `train_model.py --numpy-dir nn_inputs`
(used in every Phase 4 command below) expects each `cifar_synthetic_data/fast5/image_{i}/`
folder to already contain its own `nn_inputs/` subfolder of `.npz` chunks. Nothing above
produces this. The right tool is `main/scripts/numpy_synthetic.py` — a purpose-built
script for exactly this (distinct from `data_prepare_numpy.py`, which is for the
bacterial/Wick data in 2c). It loops over every `image_{i}` folder internally, so it's a
single call, not a bash loop, and its directory checks (`fastq/`, `fast5/`, `genomes/`
under `--path`) line up with what `dataset_generator.py` produces above:

```bash
python -m main.scripts.numpy_synthetic \
  --path ../data/cifar_synthetic_data \
  --output-dirname nn_inputs \
  --total-files 1 --window-size 2000 --window-slide 0 \
  --n-cores 8 --verbose
```
Runs in the main venv (not the `tools` conda env — no Tombo dependency here). Expect
this to take a while across thousands of image folders if you match the paper's
5000/10000-image runs.

### 2c. Bacterial data (only if redoing the benchmark pre-training itself)

`tombo_processing.py` needs the Tombo conda env (like `tombo_synthetic.py` in 2a/2b);
`create_fastqs.py` and `data_prepare_numpy.py` don't and should run in the main venv:

```bash
python main/scripts/create_fastqs.py --path <data dir> --verbose

conda activate tools
python main/scripts/tombo_processing.py --path <data dir> --verbose --processes 4
conda deactivate

python -m main.scripts.data_prepare_numpy --path <data dir> \
  --total-files 200 --window-size 2000 --window-slide 0 \
  --n-cores 8 --verbose --is-wick --out-dir nn_inputs
```

---

## Phase 3 — Benchmark pre-training (SKIP — checkpoint survives)

`checkpoint_570209.pt` already exists at `<repo>/checkpoint_570209.pt`. **Do not redo
this phase** unless you specifically need Table 3's benchmark-model profiling row —
it takes ~56 hours on a Tesla A30. For completeness, the command is:
```bash
python -m main.scripts.train_model --path <bacterial data dir> \
  --is-wick --output-dir ../experiments/finetune/models/default \
  --decoder beam_search --loss ctc \
  --num-epochs 4 --numpy-dir nn_inputs \
  --model SACallLoRA --profile
```

---

## Phase 4 — Fine-tuning runs (needed for Table 1 & Table 3)

All 5,000-image runs are 45,000 steps, batch size 64. Add `--profile` to every run —
it only profiles the first ~120 steps of epoch 0, so it's free to leave on and gives
you Table 3's `profile.csv` for no extra cost. Each produces
`checkpoints/checkpoint_45000.pt` + `log.csv` + `profile.csv` in its `--output-dir`.

```bash
CKPT=../checkpoint_570209.pt   # the surviving benchmark checkpoint
DATA=../data/cifar_synthetic_data

# Whole-Model Fine-Tuning
python -m main.scripts.train_model --path $DATA \
  --is-synthetic --output-dir ../experiments/finetune/models/5000_bacterial_benchmark_finetuned_45000 \
  --decoder beam_search --loss ctc --num-epochs 100000 --numpy-dir nn_inputs \
  --model SACallLoRA --max-steps 45000 \
  --image-start 0 --image-end 5000 \
  --checkpoint $CKPT --profile

# Decoder Fine-Tuning
python -m main.scripts.train_model --path $DATA \
  --is-synthetic --output-dir ../experiments/finetune/models/5000_images_model_decoder_finetune_45000 \
  --decoder beam_search --loss ctc --num-epochs 100000 --numpy-dir nn_inputs \
  --model SACallLoRA --max-steps 45000 \
  --image-start 0 --image-end 5000 \
  --finetune-checkpoint $CKPT --decoder-finetune --profile

# LoRA — ranks 1, 2, 4, 8, 16 (Table 1 needs all five; the recovered rank_run.sh
# script is missing rank 2 — include it here)
for RANK in 1 2 4 8 16; do
  python -m main.scripts.train_model --path $DATA \
    --is-synthetic --output-dir ../experiments/finetune/models/5000_images_model_finetune_rank_${RANK}_45000 \
    --decoder beam_search --loss ctc --num-epochs 100000 --numpy-dir nn_inputs \
    --model SACallLoRA --max-steps 45000 \
    --image-start 0 --image-end 5000 \
    --finetune-checkpoint $CKPT --lora-rank $RANK -qk --profile
done
```

---

## Phase 5 — Basecalling + evaluation

### 5a. Baselines (Tables 1, 4–9) — only needs the surviving checkpoint + Phase 2a data

```bash
mkdir -p logs
CKPT=../checkpoint_570209.pt
DATA=../data/goldman_synthetic_data

for OUTDIR in ../experiments/beam_search/default ../experiments/finetune/default; do
  python -um main.scripts.basecall \
    --path $DATA --checkpoint $CKPT --output $OUTDIR \
    --chunk-size 2000 --window-overlap 0 --batch-size 64 \
    --beam-size 3 --beam-threshold 0.0 --verbose --overwrite \
    --decoder beam_search --model SACallLoRA --is-synthetic \
    | tee -a logs/beam_search.log

  python -um main.scripts.evaluate_model \
    --reference $DATA --predictions $OUTDIR \
    --verbose --is-synthetic | tee -a logs/beam_search.log
done
```
(Both baselines are the *same underlying run* — the notebook just reads it from two
locations. Running it once and copying the folder is equivalent to running it twice.)

### 5b. Table 1's Whole-Model / Decoder / LoRA rows

```bash
DATA=../data/goldman_synthetic_data

# Whole-Model
python -um main.scripts.basecall --path $DATA \
  --checkpoint ../experiments/finetune/models/5000_bacterial_benchmark_finetuned_45000/checkpoints/checkpoint_45000.pt \
  --output ../experiments/finetune/5000_bacterial_benchmark_finetuned_45000/ \
  --chunk-size 2000 --window-overlap 0 --batch-size 64 --beam-size 3 --beam-threshold 0.0 \
  --verbose --overwrite --decoder beam_search --model SACallLoRA --is-synthetic | tee -a logs/beam_search.log
python -um main.scripts.evaluate_model --reference $DATA \
  --predictions ../experiments/finetune/5000_bacterial_benchmark_finetuned_45000/ \
  --verbose --is-synthetic | tee -a logs/beam_search.log

# Decoder
python -um main.scripts.basecall --path $DATA \
  --checkpoint ../experiments/finetune/models/5000_images_model_decoder_finetune_45000/checkpoints/checkpoint_45000.pt \
  --output ../experiments/finetune/5000_images_model_decoder_finetune_45000/ \
  --chunk-size 2000 --window-overlap 0 --batch-size 64 --beam-size 3 --beam-threshold 0.0 \
  --verbose --overwrite --decoder beam_search --model SACallLoRA --is-synthetic | tee -a logs/beam_search.log
python -um main.scripts.evaluate_model --reference $DATA \
  --predictions ../experiments/finetune/5000_images_model_decoder_finetune_45000/ \
  --verbose --is-synthetic | tee -a logs/beam_search.log

# LoRA ranks 1,2,4,8,16 — this is what rank_run.sh does, but rank_run.sh's loop
# ("1 4 8 16") is missing rank 2, so use this instead:
for RANK in 1 2 4 8 16; do
  CHECKPOINT_DIR="../experiments/finetune/models/5000_images_model_finetune_rank_${RANK}_45000/checkpoints/checkpoint_45000.pt"
  OUTDIR="../experiments/finetune/5000_images_model_finetune_rank_${RANK}_45000/"
  python -um main.scripts.basecall --path $DATA --checkpoint $CHECKPOINT_DIR --output $OUTDIR \
    --chunk-size 2000 --window-overlap 0 --batch-size 64 --beam-size 3 --beam-threshold 0.0 \
    --verbose --overwrite --decoder beam_search --model SACallLoRA --is-synthetic \
    --lora-rank $RANK -qk | tee -a logs/beam_search.log
  python -um main.scripts.evaluate_model --reference $DATA --predictions $OUTDIR \
    --verbose --is-synthetic | tee -a logs/beam_search.log
done
```

### 5c. Tables 4–9 — use the recovered scripts, with fixes

Run these from `final-project/` (`bash ../<script>.sh` if they're still sitting in
`<repo>/`, or copy them in here first):

| Table | Script | Fix needed before running |
|---|---|---|
| 4 | `redo_tables.sh` (Set 1 + Set 2) | Rename `OUTDIR="...max_repeats_${REPEATS}_length_lambda_${PENALTY}"` → `OUTDIR="...max_repeats_${REPEATS}_penalty_${PENALTY}"` in Set 1. Set 2 is fine as-is. |
| 5 | `redo_tables.sh` (Set 3) | Loop is `for LAMBDA in 1 5 10 15 20`; change to `for LAMBDA in 1 5 10 15 20 25` |
| 6 | `redo_stuff2.sh` | None — run as-is |
| 7 | `redo_stuff3.sh` | None — run as-is |
| 8 | `redo_stuff4.sh` | None — run as-is |
| 9 | `redo_kl_and_combined.sh` (Set 2, currently commented out) | Uncomment the `# --- Set 2 ---` block |

All five point at `../checkpoint_570209.pt` and `../data/goldman_synthetic_data/`, both
of which are ready after Phases 1b/2a — no training required for any of Tables 4–9.

---

## Phase 6 — Payload accuracy

`payload_accuracy_synthetic.py` needs to run against **every** result directory from
Phases 5a–5c to populate `payload_accuracy.txt` (required by every table). The
recovered `calculate_payload_accuracy.sh` only lists the 14 `finetune/`-side
directories — extend it to also cover every `beam_search/` directory:

```bash
GOLDMAN_DIR=../data/goldman_synthetic_data

INPUT_DIRS=(
  "../experiments/finetune/default"
  "../experiments/finetune/5000_bacterial_benchmark_finetuned_45000"
  "../experiments/finetune/5000_images_model_decoder_finetune_45000"
  "../experiments/finetune/5000_images_model_finetune_rank_1_45000"
  "../experiments/finetune/5000_images_model_finetune_rank_2_45000"
  "../experiments/finetune/5000_images_model_finetune_rank_4_45000"
  "../experiments/finetune/5000_images_model_finetune_rank_8_45000"
  "../experiments/finetune/5000_images_model_finetune_rank_16_45000"
  "../experiments/beam_search/default"
  "../experiments/beam_search/max_repeats_1_penalty_0.2"
  "../experiments/beam_search/max_repeats_2_penalty_0.2"
  "../experiments/beam_search/max_repeats_1_penalty_0.5"
  "../experiments/beam_search/max_repeats_2_penalty_0.5"
  "../experiments/beam_search/max_repeats_1"
  "../experiments/beam_search/length_lambda_1"
  "../experiments/beam_search/length_lambda_5"
  "../experiments/beam_search/length_lambda_10"
  "../experiments/beam_search/length_lambda_15"
  "../experiments/beam_search/length_lambda_20"
  "../experiments/beam_search/length_lambda_25"
  "../experiments/beam_search/length_discard_0.05"
  "../experiments/beam_search/length_discard_0.1"
  "../experiments/beam_search/length_discard_0.2"
  "../experiments/beam_search/length_discard_0.5"
  "../experiments/beam_search/kl_penalty_1"
  "../experiments/beam_search/kl_penalty_5"
  "../experiments/beam_search/kl_penalty_10"
  "../experiments/beam_search/kl_penalty_15"
  "../experiments/beam_search/kl_penalty_20"
  "../experiments/beam_search/kl_discard_0.01"
  "../experiments/beam_search/kl_discard_0.05"
  "../experiments/beam_search/kl_discard_0.1"
  "../experiments/beam_search/combined_no_repeats_length_lambda_5_kl_penalty_1"
  "../experiments/beam_search/combined_no_repeats_length_lambda_5_kl_penalty_5"
  "../experiments/beam_search/combined_no_repeats_length_lambda_5_kl_penalty_10"
  "../experiments/beam_search/combined_no_repeats_length_lambda_10_kl_penalty_1"
  "../experiments/beam_search/combined_no_repeats_length_lambda_10_kl_penalty_5"
  "../experiments/beam_search/combined_no_repeats_length_lambda_10_kl_penalty_10"
  "../experiments/beam_search/combined_no_repeats_length_lambda_15_kl_penalty_1"
  "../experiments/beam_search/combined_no_repeats_length_lambda_15_kl_penalty_5"
  "../experiments/beam_search/combined_no_repeats_length_lambda_15_kl_penalty_10"
)

for INPUT_DIR in "${INPUT_DIRS[@]}"; do
  echo "Running on $INPUT_DIR"
  python -m main.scripts.payload_accuracy_synthetic \
    --input-dir "$INPUT_DIR" --goldman-dir "$GOLDMAN_DIR"
done
```

---

## Phase 7 — Generate the tables

Once every directory above has `basecalls.fastq`, `basecall_stats.csv`,
`evaluation/evaluation.csv`, `evaluation/alignment_stats.csv`, and
`payload_accuracy.txt`, open `notebooks/regenerate_tables.ipynb`.

> For the multi-seed tables used in the paper, use `notebooks/regenerate_tables_multiseed.ipynb`
> instead (outputs go to `Reports/Final Report/`). See `final-project/README.md`, section
> "Reproducing the paper tables (multi-seed)". The instructions below apply only to the
> single-seed notebook.

**Do NOT use "Run All" / run top to bottom.** The notebook has 26 cells total, and only
9 of them are needed for the 9 tables — the rest are supplementary/exploratory (bacterial
consensus-accuracy comparisons, bonus KL-discard/entropy-ratio sweeps, S2/S4-appendix
data-regime tables, a "combined" and "benchmark_on_5000_images" variant that never made
the final paper). Those other cells reference directories this guide never has you
generate (`experiments/bacterial_benchmark`, `squigulated_wick_results`,
`kl_discard_*_entropy_ratio_*`, `10000_images_model_finetune_rank_4_45000`, etc.) — if
you run the notebook sequentially, it will throw `FileNotFoundError` on **cell 2** (the
very first data-dependent cell) and halt before ever reaching the cells you actually
need. Run these specific cells individually instead (Jupyter: click into each cell,
`Shift+Enter`), in this order:

1. **Cell 0** first, always — imports and `sys.path`/`os.chdir` setup everything else depends on. One edit required: `project_root = "/mnt/c/Users/Advait/Documents/Imperial/Year_4/Final_Year_Project/final-project"` — update to wherever this repo actually lives now (e.g. `E:\Documents\Imperial\Year_4\Final_Year_Project\final-project` on Windows, or the WSL-mapped equivalent).
2. **Cell 1** next, always — defines `generate_results`/`generate_latex`/helper functions every table cell calls. Doesn't touch disk itself, just defines functions.
3. Then, in any order, only the cells you need:

| Cell | Produces |
|---|---|
| 15 | **Table 1** |
| 17 | **Table 2** (no data files needed — pure code) |
| 19 | **Table 3** |
| 5 | **Table 4** |
| 6 (1st block) | **Table 5** |
| 6 (2nd block) | **Table 6** |
| 7 (1st block) | **Table 7** |
| 7 (2nd block) | **Table 8** |
| 13 | **Table 9** |

Each cell prints LaTeX table rows directly to stdout. Skip every other cell (2, 3, 4, 8–12, 14, 16, 18, 20–24) unless you've separately generated the extra data they depend on.

---

## Cost/effort summary

| Phase | Needs | Approx. cost |
|---|---|---|
| Tables 4–9 (beam search sweeps) | `checkpoint_570209.pt` (have it) + Goldman eval data (download 4 files + regenerate) | Cheap — no training, just basecalling/eval passes |
| Table 1 baseline + `beam_search/default` | Same as above | Cheap |
| Table 1 Whole-Model/Decoder/LoRA rows + Table 2 + Table 3 | CIFAR-10 + full retraining of 7 model variants, 45,000 steps each | Expensive — GPU-hours per run (paper used a Tesla A30); Table 2 alone is free (pure code, no training) |
| Benchmark pretraining (only for Table 3's "default" profiling row) | 400GB Wick dataset download + 56 hours training | Very expensive — skip unless you specifically need this one row |

If you want the cheapest path to "fully reproduced tables," do Tables 4–9 and Table 1's
baseline rows first (all unblocked today), and treat the fine-tuned-checkpoint-dependent
rows (Whole-Model/Decoder/LoRA in Tables 1 & 3) as a separate, much larger retraining
job.
