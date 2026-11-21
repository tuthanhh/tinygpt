"""Train a byte-level BPE tokenizer on a training text file.

    uv run python scripts/train_bpe.py data/vi_wiki_noaccents/train.txt --vocab-size 4096
    uv run python scripts/train_bpe.py data/vi_wiki_seg/bpe_sample.txt --vocab-size 8192 --pattern words

Writes <out-root>/<name>/{vocab.pkl, merges.pkl, meta.json}. Load it with
tinygpt.tokenizer.load_tokenizer(folder), which also restores the pre-token pattern.

--pattern gpt2  (default) the handout's GPT-2 pattern: "tổ_chức" -> "tổ", "_", "chức"
--pattern words letters joined by "_" stay one pre-token, so BPE can learn whole words.
                Uses WORD_PRETOKEN_PATTERN from tinygpt.tokenizer.bpe.train_bpe; the
                regex itself is saved in meta.json, so the folder alone rebuilds the tokenizer.
"""

import argparse
import inspect
import json
import pickle
import time
from pathlib import Path

from tinygpt.tokenizer.bpe import BpeTokenizer, train_bpe


def resolve_pattern(choice: str) -> str | None:
    """Regex for --pattern, or None for the trainer's built-in default (GPT-2)."""
    if choice == "gpt2":
        return None
    try:
        from tinygpt.tokenizer.bpe.train_bpe import WORD_PRETOKEN_PATTERN
    except ImportError:
        raise SystemExit("--pattern words: define WORD_PRETOKEN_PATTERN in tinygpt/tokenizer/bpe/train_bpe.py")
    for fn in (train_bpe, BpeTokenizer.__init__, BpeTokenizer.from_files):
        if "pattern" not in inspect.signature(fn).parameters:
            raise SystemExit(f"--pattern words: {fn.__qualname__} has no `pattern` argument yet")
    return WORD_PRETOKEN_PATTERN


def main() -> None:
    p = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    p.add_argument("train_path", type=Path, help="training text (NEVER the validation file)")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    p.add_argument("--out-root", type=Path, default=Path("data/tokenizers"))
    p.add_argument("--name", default=None, help="folder name (default: <train folder>_bpe<vocab size>)")
    p.add_argument("--pattern", choices=["gpt2", "words"], default="gpt2", help="pre-token pattern")
    args = p.parse_args()

    # Training on validation text leaks it into the vocab: merges would be tuned
    # to exactly the text you later measure loss on.
    if "valid" in args.train_path.name or "test" in args.train_path.name:
        raise SystemExit(f"refusing to train on {args.train_path}: looks like a held-out split")

    pattern = resolve_pattern(args.pattern)
    # Only pass `pattern` when set: the default path works with trainers that predate it.
    pattern_kw = {"pattern": pattern} if pattern is not None else {}

    name = args.name or f"{args.train_path.parent.name}_bpe{args.vocab_size}"
    out_dir = args.out_root / name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; delete it or pick another --name")

    start = time.perf_counter()
    vocab, merges = train_bpe(args.train_path, args.vocab_size, args.special_tokens, **pattern_kw)
    elapsed = time.perf_counter() - start

    out_dir.mkdir(parents=True)
    with open(out_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(out_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    meta = {
        "name": name,
        "train_path": str(args.train_path),
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": args.special_tokens,
        "train_seconds": round(elapsed, 1),
    }
    if pattern is not None:
        meta["pattern"] = pattern  # load_tokenizer passes it back to BpeTokenizer
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Sanity check: reload with the SAME pattern and measure compression on a slice
    # of the training text. Byte-level baseline is 1.0 bytes/token. A pattern mismatch
    # would still round-trip, but compression would drop: compare against the gpt2 run.
    tok = BpeTokenizer.from_files(out_dir / "vocab.pkl", out_dir / "merges.pkl", args.special_tokens, **pattern_kw)
    with open(args.train_path, encoding="utf-8") as f:
        sample = f.read(1_000_000)
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample, "round-trip failed"
    ratio = len(sample.encode("utf-8")) / len(ids)

    print(f"trained {len(vocab):,} tokens ({len(merges):,} merges) in {elapsed:.1f}s -> {out_dir}")
    print(f"pattern: {args.pattern}")
    print(f"compression on 1MB of train text: {ratio:.2f} bytes/token  (byte tokenizer: 1.00)")
    print("longest tokens:", sorted(vocab.values(), key=len, reverse=True)[:8])


if __name__ == "__main__":
    main()
