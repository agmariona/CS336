from collections.abc import Iterable, Iterator
from typing import Self
import regex as re
import json

PAT = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"
    r" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

class Tokenizer:
    """
    BPE Tokenizer
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ) -> None:
        self.merges = merges
        self.merge_rank = {merge: idx for idx, merge in enumerate(merges)}

        local_vocab = vocab.copy()
        inverse_vocab: dict[bytes, int] = \
            {tbytes: tid for tid, tbytes in local_vocab.items()}

        if special_tokens:
            # de-duplicate any special tokens
            special_tokens = list(dict.fromkeys(special_tokens))
            # sort by descending length
            special_tokens = sorted(special_tokens,
                key=lambda s: len(s), reverse=True)
            self.special_tokens_set = set(special_tokens)

            # add special tokens and make escaped versions for regex use
            esc_special_tokens: list[str] = []
            for tk in special_tokens:
                tk_enc = tk.encode('utf-8')
                if tk_enc not in inverse_vocab:
                    tid = len(local_vocab)
                    local_vocab[tid] = tk_enc
                    inverse_vocab[tk_enc] = tid
                esc_special_tokens.append(re.escape(tk))

            # capturing splitter
            self.special_splitter = re.compile(
                f'({"|".join(esc_special_tokens)})')
        else:
            self.special_tokens_set = set()
            self.special_splitter = None

        self.vocab = local_vocab
        self.inverse_vocab = inverse_vocab

        self.regex_pat = re.compile(PAT)

        # pretoken caching
        self.encode_cache: dict[str, tuple[int, ...]] = {}


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> Self:
        # read vocab
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        vocab: dict[int, bytes] = {}
        for record in payload:
            tid = record["id"]
            tbytes = bytes.fromhex(record["bytes_hex"])
            vocab[tid] = tbytes

        # read merges
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        merges: list[tuple[bytes, bytes]] = []
        for record in payload:
            left_byte = bytes.fromhex(record["left_byte_hex"])
            right_byte = bytes.fromhex(record["right_byte_hex"])
            merges.append((left_byte, right_byte))

        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(
        self,
        ptk: tuple[bytes, ...]
    ) -> tuple[int, ...]:
        while True:
            pairs = list(zip(ptk[:-1], ptk[1:]))
            if not pairs:
                break

            # find next merge
            best_rank = float("inf")
            best_pair = None
            for pair in pairs:
                if pair in self.merge_rank:
                    rank = self.merge_rank[pair]
                    if rank < best_rank:
                        best_pair = pair
                        best_rank = rank
            if best_pair is None:
                break

            # apply merge
            merged_bytes = b''.join(best_pair)
            new_ptk: list[bytes] = []
            i: int = 0
            while i < len(ptk):
                if i+1 < len(ptk) and (ptk[i], ptk[i+1]) == best_pair:
                    new_ptk.append(merged_bytes)
                    i += 2
                else:
                    new_ptk.append(ptk[i])
                    i += 1
            ptk = tuple(new_ptk)

        enc = tuple(self.inverse_vocab[byte] for byte in ptk)
        return enc

    def encode(
        self,
        text: str
    ) -> list[int]:
        if self.special_splitter:
            split_text = self.special_splitter.split(text)
        else:
            split_text = [text]

        text_enc: list[int] = []
        for split in split_text:
            if split == "":
                continue

            if split not in self.special_tokens_set:
                match_iter = self.regex_pat.finditer(split)
                for match in match_iter:
                    word = match.group(0)
                    if word in self.encode_cache:
                        text_enc.extend(self.encode_cache[word])
                        continue

                    bword = word.encode('utf-8')
                    ptk = tuple(bword[i:i+1] for i in range(len(bword)))

                    enc = self._encode_pretoken(ptk)
                    text_enc.extend(enc)
                    self.encode_cache[word] = enc
            else:
                # one-element tuple for special tokens
                if split in self.encode_cache:
                    text_enc.extend(self.encode_cache[split])
                    continue
                ptk = (split.encode('utf-8'),)

                enc = self._encode_pretoken(ptk)
                text_enc.extend(enc)
                self.encode_cache[split] = enc

        return text_enc


    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)


    def decode(
        self,
        ids: list[int]
    ) -> str:
        tokens = []
        for tid in ids:
            tokens.append(self.vocab[tid])
        seq = b"".join(tokens)

        return seq.decode('utf-8', errors='replace')

