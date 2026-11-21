# tinygpt

A small GPT-style language model written from scratch in PyTorch and trained on Vietnamese poetry.

This project builds on **Assignment 1 (Basics) of Stanford CS336: Language Modeling from Scratch**
([course site](https://cs336.stanford.edu), [assignment repo](https://github.com/stanford-cs336/assignment1-basics)).
It follows the assignment handout's architecture and uses its unit tests. It is a personal learning project and is not
affiliated with Stanford.

## What's inside

- **Data**: Vietnamese poetry
  ([bigscience-data/roots_vi_vietnamese_poetry](https://huggingface.co/datasets/bigscience-data/roots_vi_vietnamese_poetry),
  in `data/vietnamese_poetry/`, ~85 MB of accented training text) for the main runs;
  [hchautran/vietnamese-wiki](https://huggingface.co/datasets/hchautran/vietnamese-wiki) is also supported. The last
  ~5% of documents are held out for validation. The dataset is chosen per run in its config.

- **Model**: decoder-only Transformer with pre-norm RMSNorm, SwiGLU, RoPE, and causal multi-head attention, without
  bias terms. Default size: `d_model=256`, 4 layers, 4 heads, context 256, about 3.2M parameters.
- **Tokenizer**: byte-level (vocabulary 257: 256 bytes plus one separator token) or byte-level BPE (4096 tokens for the
  poetry corpus, about 4.8 bytes per token). A config picks one with `tokenizer_spec`: `"byte"` or a tokenizer folder.
- **Training**: SGD to validate the loop, then AdamW with linear warmup, cosine decay, and gradient clipping. Each run
  has one JSON config in `configs/` and one folder under `runs/` holding its config, metrics CSV, and checkpoints.
- **Generation**: temperature and top-p sampling from a saved checkpoint, run from the command line.

## Layout

```
src/tinygpt/       model, tokenizer, data, training, and generation modules
scripts/           thin entry points: prepare data, train a tokenizer, train, generate
configs/           one JSON config per training run
tests/             CS336 unit tests; tests/adapters.py connects them to this code
runs/              one folder per training run
data/              datasets (not committed)
```

## Setup

The project uses [`uv`](https://docs.astral.sh/uv/).

```sh
uv run pytest            # run the unit tests
```

Download the [Vietnamese Wikipedia dataset](https://huggingface.co/datasets/hchautran/vietnamese-wiki) into
`data/vi_raw/`. Then run `prepare_vi.py` to convert it to the plain-text format the training code expects: documents
separated by `<|endoftext|>`, with the last 5% held out for validation.

```sh
uv run python scripts/prepare_vi.py data/vi_raw/*.parquet --text-column <text column>
```

This writes `data/vi_train.txt` and `data/vi_valid.txt` (about 100 MB of training text by default; change with
`--train-mb`). Add `--strip-accents` to remove Vietnamese diacritics; the output files are then named `vi_noaccent_*`.

For the [poetry dataset](https://huggingface.co/datasets/bigscience-data/roots_vi_vietnamese_poetry), download its
parquet file into `data/vietnamese_poetry/` and write the text files next to it:

```sh
uv run python scripts/prepare_vi.py data/vietnamese_poetry/*.parquet --text-column text --out-dir data/vietnamese_poetry
```

## Training

Train a BPE tokenizer on the training text only. The output goes to `data/tokenizers/<corpus>_bpe<vocab size>/`:

```sh
uv run python scripts/train_bpe.py data/vietnamese_poetry/vi_train.txt --vocab-size 4096
```

Train a model from a config. The run folder defaults to `runs/<config file name>`:

```sh
uv run python scripts/train.py configs/vi_poetry.json
```

Generate from a checkpoint, interactively or once with `--prompt`:

```sh
uv run python scripts/generate.py --ckpt runs/vi_poetry/ckpt_<step>.pt --seed 0
```

## Training on Colab

`notebooks/colab_train_demo.ipynb` runs the larger configuration (`configs/vi_wiki_colab_l4.json`, about 168M
parameters, context 256, TF32 matmuls, sized to fill about 80% of the GPU's memory) on a Colab L4 GPU; `configs/vi_wiki_colab.json` is a smaller T4 variant. It uses
the word-segmented Vietnamese Wikipedia
([hchautran/vietnamese-wiki-segmented](https://huggingface.co/datasets/hchautran/vietnamese-wiki-segmented): words
joined by underscores, punctuation removed). The notebook writes about 1 GB of training text to `data/vi_wiki_seg/`
(`prepare_vi.py --segmented keep`, or `join` to turn underscores back into spaces), trains an 8192-token BPE tokenizer
on a 100 MB sample (`train_bpe.py --pattern words` keeps underscore-joined words as one pre-token; the pattern is saved
in the tokenizer's `meta.json`), tokenizes the text, runs a 200-step speed test, trains, plots the loss curves, and serves a Gradio
demo. The tokenizer, token cache, and run folders are kept on Google Drive so they survive disconnects.

To tokenize locally instead of on Colab, run the command below and upload the two `.npy` files from `data/cache/` to
`MyDrive/tinygpt/cache/` (and the tokenizer folder to `MyDrive/tinygpt/tokenizers/`):

```sh
uv run python scripts/tokenize_data.py configs/vi_wiki_colab.json
```

## Acknowledgements

The tests, adapter interface, and model specification come from the
[CS336 Assignment 1 starter code](https://github.com/stanford-cs336/assignment1-basics) by the Stanford CS336 staff.
