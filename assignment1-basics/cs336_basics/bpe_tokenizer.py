import regex as re
from .pretokenization_example import find_chunk_boundaries

def bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Serial BPE tokenizer
    """
    # de-duplicate any special tokens
    special_tokens = list(dict.fromkeys(special_tokens))

    # ensure vocab size is big enough
    if vocab_size < (len(special_tokens) + 256):
        raise ValueError(
            "vocab_size must be at least 256 + len(special_tokens)"
        )

    # initialize vocabulary with all byte values
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    esc_special_tokens: list[str] = []
    # add special tokens and make escaped versions for regex use
    for tk in special_tokens:
        vocab[len(vocab)] = tk.encode("utf-8")
        esc_special_tokens.append(re.escape(tk))

    # pretokenize in chunks
    PAT = (
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"
        r" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    )
    pretoken_counts: dict[tuple[bytes, ...], int] = {}
    with open(input_path, "rb") as f:
        num_chunks = 4
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            if len(esc_special_tokens) > 0:
                split_chunks = re.split("|".join(esc_special_tokens), chunk)
            else:
                split_chunks = [chunk]

            for split in split_chunks:
                match_iter = re.finditer(PAT, split)
                for match in match_iter:
                    word = match.group(0)
                    bword = word.encode('utf-8')
                    ptk = tuple(bword[i:i+1] for i in range(len(bword)))
                    # if ptk in pretoken_counts:
                    #     pretoken_counts[ptk] += 1
                    # else:
                    #     pretoken_counts[ptk] = 1
                    pretoken_counts = inc_count(ptk, pretoken_counts, 1)

    # merge
    # first, count up pairs using base vocab
    merge_hist: list[tuple[bytes, bytes]] = []
    pair_counts: dict[tuple[bytes, bytes], int] = \
        compute_pair_counts(pretoken_counts)

    # now merge until vocab upper limit or until no more merging possible
    while (len(vocab) < vocab_size) and (len(pair_counts) > 0):

        # DEBUG #
        # print('\n')
        # for k,v in pretoken_counts.items():
        #     print(k, ':', v)
        # print('\n')
        # for k,v in pair_counts.items():
        #     print(k, ':', v)
        ###

        # find max pair
        best_pair, best_count = \
            max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))

        # add new vocab item
        merged_bytes = b''.join(best_pair)
        vocab[len(vocab)] = merged_bytes
        merge_hist.append(best_pair)

        new_pretoken_counts: dict[tuple[bytes, ...], int] = {}
        for pretoken, count in pretoken_counts.items():
            new_ptk: list[bytes] = []

            i = 0
            while i < len(pretoken):
                curr_pair = pretoken[i:i+2]
                if curr_pair == best_pair:
                    new_ptk.append(merged_bytes)
                    i += 2
                else:
                    new_ptk.append(pretoken[i])
                    i += 1

            new_pretoken: tuple[bytes, ...] = tuple(new_ptk)
            new_pretoken_counts = \
                inc_count(new_pretoken, new_pretoken_counts, count)
        pretoken_counts = new_pretoken_counts

        # recompute pair counts naively
        pair_counts = compute_pair_counts(pretoken_counts)

    return (vocab, merge_hist)

def compute_pair_counts(
    pretoken_counts: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, bytes], int]:
    """
    helper function for unoptimized merge step
    """

    pair_counts: dict[tuple[bytes, bytes], int] = {}
    for pretoken, count in pretoken_counts.items():
        if len(pretoken) < 2:
            continue

        for i in range(len(pretoken)-1):
            pair = pretoken[i:i+2]
            # if pair in pair_counts:
            #     pair_counts[pair] += count
            # else:
            #     pair_counts[pair] = count
            pair_counts = inc_count(pair, pair_counts, count)

    return pair_counts


def inc_count(
    key: tuple[bytes, ...],
    counts: dict[tuple[bytes, ...], int],
    inc: int
) -> dict[tuple[bytes, ...], int]:
    """
    helper function to initialize or increment key counts
    """
    if key in counts:
        counts[key] += inc
    else:
        counts[key] = inc
    return counts
