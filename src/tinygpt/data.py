from pathlib import Path

import numpy as np
from tinygpt.tokenizer.byte import EOT_TOKEN
import torch
import numpy.typing as npt


def read_text(path: str | Path, max_bytes: int | None = None) -> str:
    """Read the whole file, or the first max_bytes cut back to the last <|endoftext|>,
    so no story is split in half."""
    with open(path, "rb") as f:
        raw = f.read() if max_bytes is None else f.read(max_bytes)

    if max_bytes is not None:
        marker = EOT_TOKEN.encode("utf-8")
        cut = raw.rfind(marker)
        if cut == -1:
            raise ValueError(f"no {EOT_TOKEN} in the first {max_bytes} bytes of {path}")
        raw = raw[: cut + len(marker)]  # keep the separator that ends the last story

    # The cut is right after an ASCII marker, so it never splits a multi-byte character.
    return raw.decode("utf-8")


def encode_to_array(text: str, tokenizer, docs_per_chunk: int = 10_000) -> npt.NDArray[np.uint16]:
    """Encode to one flat uint16 array, a chunk of stories at a time.

    Encoding 100MB in one call builds a ~100M-element Python list (~800MB of
    pointers) before numpy sees it; chunking keeps the temporary small.
    Stories are rejoined with the separator, so the result equals encode(text).
    """
    assert tokenizer.vocab_size <= 2**16, "uint16 holds IDs up to 65535"
    docs = text.split(EOT_TOKEN)
    chunks = []
    for start in range(0, len(docs), docs_per_chunk):
        piece = EOT_TOKEN.join(docs[start : start + docs_per_chunk])
        if start > 0:
            piece = EOT_TOKEN + piece  # the separator between this chunk and the previous one
        chunks.append(np.array(tokenizer.encode(piece), dtype=np.uint16))
    return np.concatenate(chunks)


def load_tokens(
    txt_path: str | Path,
    tokenizer,
    max_bytes: int | None = None,
    cache_dir: str | Path | None = None,
) -> npt.NDArray[np.uint16]:
    """Tokens for a text file, cached as .npy so later runs skip encoding."""

    size_tag = "full" if max_bytes is None else f"{max_bytes // 1_000_000}MB"
    cache_path = (
        Path(cache_dir) / f"{Path(txt_path).stem}_{tokenizer.name}_{size_tag}.npy" if cache_dir is not None else None
    )

    if cache_path is not None and Path(cache_path).exists():
        return np.load(cache_path, mmap_mode="r")
    tokens = encode_to_array(read_text(txt_path, max_bytes), tokenizer)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, tokens)
    return tokens


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows: x = dataset[s : s+L], y = dataset[s+1 : s+1+L].
    Returns two LongTensors of shape (batch_size, context_length) on device."""
    n = len(dataset)
    assert n > context_length, f"dataset ({n} tokens) too short for context_length {context_length}"

    # Largest valid start is n - context_length - 1: then y's last index is n - 1.
    # randint's high is exclusive, so high = n - context_length.
    starts = np.random.randint(0, n - context_length, size=batch_size)

    # (batch, 1) + (L,) -> (batch, L) index grid; one fancy-index gathers every window.
    idx = starts[:, None] + np.arange(context_length)
    x = torch.from_numpy(dataset[idx].astype(np.int64))
    y = torch.from_numpy(dataset[idx + 1].astype(np.int64))
    return x.to(device), y.to(device)
