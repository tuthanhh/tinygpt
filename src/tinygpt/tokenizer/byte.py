EOT_TOKEN = "<|endoftext|>"


class ByteTokenizer:
    name = "byte_tokenizer"

    def __init__(self) -> None:
        self.eot_id = 256
        self.vocab_size = 257

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for i, piece in enumerate(text.split(EOT_TOKEN)):
            if i > 0:
                ids.append(self.eot_id)
            ids.extend(piece.encode("utf-8"))
        return ids

    def decode(self, ids: list[int]) -> str:

        eot_bytes = EOT_TOKEN.encode("utf-8")
        buf = bytearray()

        for i in ids:
            if i == self.eot_id:
                buf.extend(eot_bytes)
            elif 0 <= i < 256:
                buf.append(i)
            else:
                raise ValueError(f"token id {i} out of range for vocab_size {self.vocab_size}")
        return buf.decode("utf-8", errors="replace")
