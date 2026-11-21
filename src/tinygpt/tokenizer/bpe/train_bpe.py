import mmap
import multiprocessing as mp
import os
from collections import Counter
from itertools import pairwise
from operator import itemgetter
from typing import BinaryIO, cast

import regex as re

# Lookup table mapping integer byte value to bytes type
BYTE_TOKENS = tuple(bytes([i]) for i in range(256))
# This pattern is tuned to English, splitting out contractions, words, numbers, and others
PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# Same, except letters joined by "_" stay one pre-token, for word-segmented text
# ("tổ_chức quốc_tế" -> "tổ_chức", " quốc_tế"), so BPE can learn whole words.
# Only letters join: "abc_123" -> "abc", "_", "123"; a stray "_" falls to the symbol branch.
WORD_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+(?:_\p{L}+)*| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def build_delim_regex(special_tokens: list[bytes]) -> bytes:
    """
    Build an alternation of special tokens, longest first.

    Args:
        special_tokens (list[bytes]): A list of special tokens.

    Returns:
        A regex bytes pattern of special tokens alternation.
    """

    # For typing purpose, so that the sorted call get infered as list[bytes]
    def bytes_len(token: bytes) -> int:
        return len(token)

    tokens_sorted = sorted(special_tokens, key=bytes_len, reverse=True)
    escaped = [re.escape(token) for token in tokens_sorted]
    return b"|".join(escaped)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    Return at most `desired_num_chunks` chunks.

    Args:
        file (BinaryIO): The corpus file object, as open by `open(..., "rb")`.
        desired_num_chunks (int): The amount of chunks to split into.
        split_special_tokens (list[bytes]): The non-empty boundary tokens for chunking.
            If the list is empty, only a single chunk will be produced.
            A chunk always start at one of these tokens except for the potentially the first chunk.
    """
    assert isinstance(split_special_tokens, list) and all(
        isinstance(token, bytes) and token != b"" for token in split_special_tokens
    ), "Must represent special tokens as a list of non-empty bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # Edge case
    if file_size == 0 or len(split_special_tokens) == 0:
        return [0, file_size]

    chunk_size = max(1, file_size // max(1, desired_num_chunks))

    delim_pattern = build_delim_regex(split_special_tokens)
    boundaries = [0]

    with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for match in re.finditer(delim_pattern, mm):
            if match.start() >= boundaries[-1] + chunk_size:
                boundaries.append(match.start())

    boundaries.append(file_size)
    return boundaries


def count_pretoken(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[bytes],
    pattern: str = PRETOKEN_PATTERN,
) -> Counter[bytes]:
    """
    Split the chunks [start, end) of the corpus into pretokens and count them.

    Args:
        input_path (str | os.PathLike): Path to the tokenizer training data.
        start (int): Index of the start bytes of the chunk.
        end (int): Index of the one-past-end bytes of the chunk.
        special_tokens (list[bytes]): A list of string special tokens.
            These strings act as boundary for pretokens, and are not counted as pretokens themselves.
        pattern (str): Pre-token regex. Passed explicitly (not read from a global) because
            this runs in worker processes.

    Returns:
        Counter[bytes]: a dictionary of pretokens and their corresponding count.
    """

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start)

    if len(special_tokens) != 0:
        # Special tokens are boundaries and not pretokens themselves
        samples = cast(list[bytes], re.split(build_delim_regex(special_tokens), chunk))
    else:
        samples = [chunk]

    count = Counter[bytes]()
    for sample in samples:
        for match in re.finditer(pattern, sample.decode(errors="ignore")):
            count[match.group().encode()] += 1

    return count


def pretokenize_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    pattern: str = PRETOKEN_PATTERN,
) -> list[tuple[tuple[bytes, ...], int]]:
    """
    Given the path to an input corpus, split the corpus into pretokens and count them.

    Args:
        input_path (str | os.PathLike): Path to tokenizer training data.
        special_tokens (list[str]): A list of string special tokens.
            These strings act as boundary for pretokens, and are not counted as pretokens themselves.
        pattern (str): Pre-token regex (default: GPT-2 style PRETOKEN_PATTERN).

    Returns:
        list[tuple[tuple[bytes, ...], int]]:
            A list of pretokens and their corresponding counts.
    """

    # Config (to be refactored)
    num_processes = 4
    chunks_per_process = 4

    # Chunking for parallel processing speedup
    special_tokens_bytes = [token.encode() for token in special_tokens]

    with open(input_path, "rb") as file:
        chunk_boundaries = find_chunk_boundaries(file, num_processes * chunks_per_process, special_tokens_bytes)

    # Counting each chunks is independent works
    with mp.Pool(num_processes) as pool:
        pretoken_counts = pool.starmap(
            count_pretoken,
            [
                (input_path, start, end, special_tokens_bytes, pattern)
                for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])
            ],
        )

    pretoken_count = Counter[bytes]()
    for count in pretoken_counts:
        pretoken_count += count

    pretokens = [
        (tuple([BYTE_TOKENS[byte] for byte in pretoken]), pretoken_count[pretoken]) for pretoken in pretoken_count
    ]

    return pretokens


def index_byte_pairs(
    pretokens: list[tuple[tuple[bytes, ...], int]],
) -> tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[int]]]:
    """
    Given pretokens information, counts byte-level pairs and index their pretoken origins.

    Args:
        pretokens (list[tuple[tuple[bytes, ...], int]]): A list of pretokens and their corresponding counts.

    Returns:
        tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[int]]]:
            pair_count:
                Counting statistics for each byte-level pair in the given pretoken list.
            pair_pretoken_indices:
                Indices to the pretokens in which the given pair appears.
    """

    pair_count = Counter[tuple[bytes, bytes]]()
    pair_pretoken_indices = dict[tuple[bytes, bytes], set[int]]()

    for index, (pretoken, count) in enumerate(pretokens):
        for pair in pairwise(pretoken):
            pair_count[pair] += count
            pair_pretoken_indices.setdefault(pair, set()).add(index)

    return (pair_count, pair_pretoken_indices)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    pattern: str = PRETOKEN_PATTERN,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given the path to an input corpus, train a BPE tokenizer and output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be kept as a single token.
        pattern (str): Pre-token regex. Encoding must use the same one (BpeTokenizer(pattern=...)).

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
            merges:
                BPE merges.
                Each list item is a tuple of bytes `(token1, token2)`, representing that `token1` was merged with `token2`.
                Merges are ordered by order of creation.
    """

    pretokens = pretokenize_file(input_path, special_tokens, pattern)
    pair_count, pair_pretoken_indices = index_byte_pairs(pretokens)

    vocab = {byte: BYTE_TOKENS[byte] for byte in range(256)}
    merges = list[tuple[bytes, bytes]]()

    def update_pair_statistics(
        old_pair: tuple[bytes, bytes],
        new_pair: tuple[bytes, bytes],
        pretoken_index: int,
        pretoken_count: int,
    ) -> None:
        pair_count[old_pair] -= pretoken_count
        if pair_count[old_pair] == 0:
            pair_count.pop(old_pair)

        pair_count[new_pair] += pretoken_count
        pair_pretoken_indices.setdefault(new_pair, set()).add(pretoken_index)

    while len(vocab) + len(special_tokens) < vocab_size and len(pair_count) > 0:
        # In case of tie, the one with larger lexicographical order wins.
        merging_pair = max(pair_count.items(), key=itemgetter(1, 0))[0]
        new_token = merging_pair[0] + merging_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(merging_pair)

        # Fix affected pretoken statistics
        for index in pair_pretoken_indices[merging_pair]:
            # Count is invariant under token merging
            count = pretokens[index][1]

            i = 0
            while i < len(pretokens[index][0]) - 1:
                # Pretoken is replaced while merging, so indirect reading is needed to avoid stale value
                pretoken = pretokens[index][0]
                # Search for the merging pair in a corresponding pretoken
                if pretoken[i] == merging_pair[0] and pretoken[i + 1] == merging_pair[1]:
                    if i != 0:
                        # There is an overlapping pair (pred, fst) preceding the merging pair (fst, snd).
                        precede_token = pretoken[i - 1]
                        old_pair = (precede_token, merging_pair[0])
                        new_pair = (precede_token, new_token)
                        update_pair_statistics(old_pair, new_pair, index, count)
                    if i != len(pretoken) - 2:
                        # There is an overlapping pair (snd, succ) succeeding the merging pair (fst, snd).
                        succeeding_token = pretoken[i + 2]
                        old_pair = (merging_pair[1], succeeding_token)
                        new_pair = (new_token, succeeding_token)
                        update_pair_statistics(old_pair, new_pair, index, count)
                    # Merge the token pair in the pretoken
                    pretokens[index] = (pretoken[:i] + (new_token,) + pretoken[i + 2 :], count)
                i += 1

        # After merging, the merging pair occurred nowhere, so its statistics must be removed
        pair_count.pop(merging_pair)

    # Account for special tokens
    for token in special_tokens:
        vocab[len(vocab)] = token.encode()

    return (vocab, merges)
