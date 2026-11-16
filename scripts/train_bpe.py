"""Train a byte-level BPE tokenizer on a training text file.

    uv run python scripts/train_bpe.py data/vi_wiki_noaccents/train.txt --vocab-size 4096

Writes <out-root>/<name>/{vocab.pkl, merges.pkl, meta.json}, loadable with
BpeTokenizer.from_files(dir / "vocab.pkl", dir / "merges.pkl", special_tokens).
"""

import argparse
import json
import pickle
import time
from pathlib import Path

# adjust to wherever your BPE trainer lives; expected to return (vocab, merges)
from tinygpt.tokenizer.bpe import BpeTokenizer, train_bpe

def main() -> None:
    p = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    p.add_argument("train_path", type=Path, help="training text (NEVER the validation file)")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    p.add_argument("--out-root", type=Path, default=Path("data/tokenizers"))
    p.add_argument("--name", default=None, help="folder name (default: default_name(...))")
    args = p.parse_args()

    # Training on validation text leaks it into the vocab: merges would be tuned
    # to exactly the text you later measure loss on.
    if "valid" in args.train_path.name or "test" in args.train_path.name:
        raise SystemExit(f"refusing to train on {args.train_path}: looks like a held-out split")

    name = args.name or f"{args.train_path.parent.name}_bpe{args.vocab_size}"
    out_dir = args.out_root / name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; delete it or pick another --name")

    start = time.perf_counter()
    vocab, merges = train_bpe(args.train_path, args.vocab_size, args.special_tokens)
    elapsed = time.perf_counter() - start

    out_dir.mkdir(parents=True)
    with open(out_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(out_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "name": name,
                "train_path": str(args.train_path),
                "vocab_size": len(vocab),
                "num_merges": len(merges),
                "special_tokens": args.special_tokens,
                "train_seconds": round(elapsed, 1),
            },
            f,
            indent=2,
        )

    # Sanity check: reload through the real loader and measure compression on
    # a slice of the training text. Byte-level baseline is 1.0 bytes/token.
    tok = BpeTokenizer.from_files(out_dir / "vocab.pkl", out_dir / "merges.pkl", args.special_tokens)
    with open(args.train_path, encoding="utf-8") as f:
        sample = f.read(1_000_000)
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample, "round-trip failed"
    ratio = len(sample.encode("utf-8")) / len(ids)

    print(f"trained {len(vocab):,} tokens ({len(merges):,} merges) in {elapsed:.1f}s -> {out_dir}")
    print(f"compression on 1MB of train text: {ratio:.2f} bytes/token  (byte tokenizer: 1.00)")
    print("longest tokens:", sorted(vocab.values(), key=len, reverse=True)[:8])


if __name__ == "__main__":
    main()
