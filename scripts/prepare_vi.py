"""Turn parquet/jsonl documents into the .txt contract the training code expects:
documents separated by <|endoftext|>.

    uv run python scripts/prepare_vi.py data/vi_raw/*.parquet --text-column story
    uv run python scripts/prepare_vi.py data/vi_raw/*.parquet --text-column story --strip-accents
"""

import argparse
import json
import random
import unicodedata
from collections.abc import Iterator
from pathlib import Path

EOT = "<|endoftext|>"


# ---------------------------------------------------------------- 1. load
def iter_documents(paths: list[Path], text_column: str) -> Iterator[str]:
    """One document per row, streamed so a multi-GB source never sits in RAM."""
    for path in paths:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(columns=[text_column], batch_size=10_000):
                for text in batch.column(0).to_pylist():
                    if text is not None:
                        yield text
        elif path.suffix in (".jsonl", ".json"):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        text = json.loads(line).get(text_column)
                        if text is not None:
                            yield text
        else:
            raise SystemExit(f"unsupported file type: {path} (use .parquet or .jsonl)")


# ---------------------------------------------------------------- 2. clean
def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    no_marks = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return no_marks.replace("đ", "d").replace("Đ", "D")  # đ is its own letter, not d + mark


def clean(text: str, min_chars: int, no_accents: bool) -> str | None:
    """Cleaned document, or None if it should be dropped."""
    text = unicodedata.normalize("NFC", text)  # one byte form per accented letter
    text = text.replace(EOT, " ")  # a stray separator would split the document
    text = text.strip()
    if no_accents:
        text = strip_accents(text)
    return text if len(text) >= min_chars else None


# ---------------------------------------------------------------- 4. write
def write_docs(path: Path, docs: list[str]) -> int:
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc + "\n" + EOT + "\n")
    return path.stat().st_size


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Vietnamese text for training.")
    p.add_argument("inputs", nargs="+", type=Path, help=".parquet or .jsonl files")
    p.add_argument("--text-column", default="story")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--name", default=None, help="file prefix (default: vi or vi_noaccent)")
    p.add_argument("--train-mb", type=float, default=100.0, help="target train file size")
    p.add_argument("--valid-frac", type=float, default=0.05)
    p.add_argument("--min-chars", type=int, default=50)
    p.add_argument("--strip-accents", action="store_true")
    args = p.parse_args()

    name = args.name or ("vi_noaccent" if args.strip_accents else "vi")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Collect just enough documents for train + valid, so the split stays ~valid_frac
    # and the train file lands near --train-mb. Sizes counted in UTF-8 bytes, which
    # is what the file (and the byte tokenizer) actually sees.
    budget = args.train_mb * 2**20 / (1 - args.valid_frac)
    docs: list[str] = []
    total_bytes = 0
    seen = dropped = 0
    for raw in iter_documents(args.inputs, args.text_column):
        seen += 1
        doc = clean(raw, args.min_chars, args.strip_accents)
        if doc is None:
            dropped += 1
            continue
        docs.append(doc)
        total_bytes += len(doc.encode("utf-8")) + len(EOT) + 2  # + separator and newlines
        if total_bytes >= budget:
            break

    if len(docs) < 2:
        raise SystemExit("too few documents after cleaning; check --text-column and inputs")

    # ------------------------------------------------------------ 3. split
    # By whole documents, never inside one: the last valid_frac go to validation.
    n_valid = max(1, round(len(docs) * args.valid_frac))
    train_docs, valid_docs = docs[:-n_valid], docs[-n_valid:]

    train_path = args.out_dir / f"{name}_train.txt"
    valid_path = args.out_dir / f"{name}_valid.txt"
    train_size = write_docs(train_path, train_docs)
    valid_size = write_docs(valid_path, valid_docs)

    # ------------------------------------------------------------ checks
    print(f"read {seen:,} rows, dropped {dropped:,} (< {args.min_chars} chars)")
    print(f"train: {len(train_docs):,} docs, {train_size / 2**20:.1f} MB -> {train_path}")
    print(f"valid: {len(valid_docs):,} docs, {valid_size / 2**20:.1f} MB -> {valid_path}")

    rng = random.Random(0)
    sample = rng.sample(docs, k=min(200, len(docs)))
    nfc_ok = all(unicodedata.is_normalized("NFC", d) for d in sample)
    n_bytes = sum(len(d.encode("utf-8")) for d in sample)
    n_chars = sum(len(d) for d in sample)
    ratio = n_bytes / n_chars
    print(f"NFC on {len(sample)} random docs: {nfc_ok}")
    print(f"bytes per char: {ratio:.3f}  ", end="")
    if args.strip_accents:
        print("(expect 1.00: pure ASCII)" if ratio < 1.01 else "(!! expected 1.00 with --strip-accents)")
    else:
        print("(expect ~1.3-1.5)" if ratio > 1.15 else "(!! near 1.0: is this unaccented text?)")

    for d in rng.sample(docs, k=min(3, len(docs))):
        print("-" * 72)
        print(d[:400] + ("..." if len(d) > 400 else ""))
    print("-" * 72)


if __name__ == "__main__":
    main()
