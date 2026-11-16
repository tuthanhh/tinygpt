import json
from pathlib import Path

from tinygpt.tokenizer.bpe import BpeTokenizer
from tinygpt.tokenizer.byte import EOT_TOKEN, ByteTokenizer


def load_tokenizer(spec: str) -> ByteTokenizer | BpeTokenizer:
    """Build the tokenizer a config or checkpoint names. Training and generation
    both go through here, so they always agree on the tokenizer.

    spec is "byte", or a folder written by scripts/train_bpe.py
    (vocab.pkl, merges.pkl, meta.json).
    """
    if spec in ("byte", ByteTokenizer.name):  # ByteTokenizer.name: checkpoints saved before specs existed
        return ByteTokenizer()

    folder = Path(spec)
    if not folder.is_dir():
        raise ValueError(f"tokenizer {spec!r}: expected 'byte' or a folder from scripts/train_bpe.py")
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    tok = BpeTokenizer.from_files(folder / "vocab.pkl", folder / "merges.pkl", meta.get("special_tokens", [EOT_TOKEN]))
    # The name keys the token cache in load_tokens: it must differ between tokenizers.
    tok.name = meta.get("name", folder.name)
    # Not checked inside BpeTokenizer: the handout tests build it without special tokens.
    if tok.eot_id is None:
        raise ValueError(f"tokenizer {spec!r} has no {EOT_TOKEN}: needed as document separator and stop token")
    return tok
