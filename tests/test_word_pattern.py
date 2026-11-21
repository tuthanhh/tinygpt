"""Whole-word pre-tokenization for word-segmented text ("tổ_chức quốc_tế").

    uv run pytest tests/test_word_pattern.py
"""

import json
import pickle

import pytest
import regex

from tinygpt.tokenizer import load_tokenizer
from tinygpt.tokenizer.bpe import BpeTokenizer, train_bpe
from tinygpt.tokenizer.bpe.train_bpe import PRETOKEN_PATTERN, WORD_PRETOKEN_PATTERN

EOT = "<|endoftext|>"


def split(pattern: str, text: str) -> list[str]:
    return regex.findall(pattern, text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tổ_chức quốc_tế", ["tổ_chức", " quốc_tế"]),
        ("Thái_Bình_Dương", ["Thái_Bình_Dương"]),
        ("là", ["là"]),
        ("abc_123", ["abc", "_", "123"]),  # digits never join
        ("_abc", ["_", "abc"]),  # stray underscore: its own piece
        ("abc_", ["abc", "_"]),
        ("a__b", ["a", "__", "b"]),
        ("năm 2020 ở Hà_Nội", ["năm", " 2020", " ở", " Hà_Nội"]),
    ],
)
def test_word_pattern_splits(text, expected):
    assert split(WORD_PRETOKEN_PATTERN, text) == expected
    assert "".join(expected) == text  # pre-tokens must cover the text exactly


def test_default_pattern_unchanged():
    assert split(PRETOKEN_PATTERN, "tổ_chức quốc_tế") == ["tổ", "_", "chức", " quốc", "_", "tế"]


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "train.txt"
    doc = " ".join(["tổ_chức quốc_tế"] * 50)
    path.write_text(f"{doc}{EOT}{doc}", encoding="utf-8")
    return path


VOCAB_SIZE = 256 + 1 + 60  # bytes + EOT + enough merges for both words


def test_train_learns_whole_words_only_with_word_pattern(corpus):
    vocab_words, _ = train_bpe(corpus, VOCAB_SIZE, [EOT], pattern=WORD_PRETOKEN_PATTERN)
    assert " quốc_tế".encode() in vocab_words.values()

    vocab_default, _ = train_bpe(corpus, VOCAB_SIZE, [EOT])
    assert not any(b"_" in t and len(t) > 1 for t in vocab_default.values())


def test_tokenizer_encodes_with_its_pattern(corpus):
    vocab, merges = train_bpe(corpus, VOCAB_SIZE, [EOT], pattern=WORD_PRETOKEN_PATTERN)
    tok = BpeTokenizer(vocab, merges, [EOT], pattern=WORD_PRETOKEN_PATTERN)
    text = f"tổ_chức quốc_tế{EOT}quốc_tế abc_123"
    assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode(" quốc_tế")) == 1

    # Same merges, default pattern: still round-trips, but cannot use the whole-word merges.
    mismatched = BpeTokenizer(vocab, merges, [EOT])
    assert mismatched.decode(mismatched.encode(text)) == text
    assert len(mismatched.encode(" quốc_tế")) > 1


def test_load_tokenizer_restores_pattern(corpus, tmp_path):
    vocab, merges = train_bpe(corpus, VOCAB_SIZE, [EOT], pattern=WORD_PRETOKEN_PATTERN)
    folder = tmp_path / "tok"
    folder.mkdir()
    (folder / "vocab.pkl").write_bytes(pickle.dumps(vocab))
    (folder / "merges.pkl").write_bytes(pickle.dumps(merges))
    meta = {"name": "test_words", "special_tokens": [EOT], "pattern": WORD_PRETOKEN_PATTERN}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    tok = load_tokenizer(str(folder))
    assert tok.pattern == WORD_PRETOKEN_PATTERN
    assert len(tok.encode(" quốc_tế")) == 1
