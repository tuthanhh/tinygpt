"""Tokenize a config's train/val text into the .npy cache, without training.

    uv run python scripts/tokenize_data.py configs/vi_wiki_colab_l4.json

train() would do the same on its first run, but encoding ~1GB takes a while, so it
is worth doing once, on its own, and copying the cache (Colab: to Google Drive).
The cache file names come from tinygpt.data.load_tokens; a cached file is reused.
"""

import argparse
import json
import time
from pathlib import Path

from tinygpt.data import load_tokens
from tinygpt.tokenizer import load_tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Build the token cache for a run config.")
    p.add_argument("config", type=Path)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    tokenizer = load_tokenizer(cfg.get("tokenizer_spec", "byte"))
    cache_dir = cfg.get("cache_dir", "data/cache")

    for key, max_bytes in (("train_path", cfg.get("train_max_bytes")), ("val_path", None)):
        path = Path(cfg[key])
        t0 = time.perf_counter()
        tokens = load_tokens(path, tokenizer, max_bytes=max_bytes, cache_dir=cache_dir)
        dt = time.perf_counter() - t0
        line = f"{key}: {len(tokens):,} tokens in {dt:.1f}s"
        if path.exists() and max_bytes is None and dt > 1.0:  # a real full encode, not a cache hit
            mb = path.stat().st_size / 2**20
            line += f"  ({mb / dt:.2f} MB/s, {path.stat().st_size / len(tokens):.2f} bytes/token)"
        print(line)


if __name__ == "__main__":
    main()
