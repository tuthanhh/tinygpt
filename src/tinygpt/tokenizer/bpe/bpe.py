import pickle
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex  # not `re`: the pattern needs \p{L} / \p{N} Unicode classes

from tinygpt.tokenizer.bpe.train_bpe import PRETOKEN_PATTERN
from tinygpt.tokenizer.byte import EOT_TOKEN

# Same pattern training used: encoding must split text exactly as training did.
PAT = regex.compile(PRETOKEN_PATTERN)


class BpeTokenizer:
    """Byte-level BPE: pre-tokenize with PAT, then apply merges inside each pre-token."""

    name = "bpe_tokenizer"

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.special_tokens = list(special_tokens or [])

        # Special tokens must have IDs; add any the vocab is missing.
        known = set(self.vocab.values())
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in known:
                self.vocab[len(self.vocab)] = b
                known.add(b)

        self.token_to_id = {b: i for i, b in self.vocab.items()}
        # Rank = position in the merge list: lower rank was learned earlier and wins.
        self.merge_rank = {pair: r for r, pair in enumerate(merges)}

        self.vocab_size = len(self.vocab)
        self.eot_id = self.token_to_id.get(EOT_TOKEN.encode("utf-8"))

        # Split on specials, LONGEST first, so an overlapping shorter special
        # (e.g. "<|endoftext|>" inside "<|endoftext|><|endoftext|>") can't win.
        if self.special_tokens:
            alternation = "|".join(
                regex.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)
            )
            self._special_re = regex.compile(f"({alternation})")  # capture: keep the specials
        else:
            self._special_re = None

        self._cache: dict[str, list[int]] = {}  # pre-token -> ids; text repeats words a lot

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "BpeTokenizer":
        """Load the pickles scripts/train_bpe.py writes. Prefer tinygpt.tokenizer.load_tokenizer,
        which also sets the name and checks for the separator token."""
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    # ------------------------------------------------------------ encoding
    def _merge_pretoken(self, pretoken: str) -> list[int]:
        cached = self._cache.get(pretoken)
        if cached is not None:
            return cached

        parts = [bytes([b]) for b in pretoken.encode("utf-8")]
        while len(parts) > 1:
            # The adjacent pair that was learned earliest, if any pair is mergeable.
            best_rank, best_i = None, -1
            for i in range(len(parts) - 1):
                r = self.merge_rank.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            pair = (parts[best_i], parts[best_i + 1])

            # Merge every occurrence of that pair, left to right, non-overlapping.
            merged: list[bytes] = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == pair:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged

        ids = [self.token_to_id[p] for p in parts]
        self._cache[pretoken] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        chunks = self._special_re.split(text) if self._special_re else [text]
        specials = set(self.special_tokens)
        ids: list[int] = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in specials:
                ids.append(self.token_to_id[chunk.encode("utf-8")])
                continue
            for m in PAT.finditer(chunk):
                ids.extend(self._merge_pretoken(m.group()))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode a stream of strings (e.g. a file's lines) without loading it all."""
        for chunk in iterable:
            yield from self.encode(chunk)

    # ------------------------------------------------------------ decoding
    def token_bytes(self, token_id: int) -> bytes:
        """Raw bytes of one token, for streaming decode in the CLI."""
        return self.vocab[token_id]

    def decode(self, ids: list[int]) -> str:
        # Join all bytes first, decode once: a character may span several tokens.
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")
