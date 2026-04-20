import regex as re
from multiprocessing import Pool
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

    # add special tokens and make escaped versions for regex use
    esc_special_tokens: list[str] = []
    for tk in special_tokens:
        vocab[len(vocab)] = tk.encode("utf-8")
        esc_special_tokens.append(re.escape(tk))
    if len(esc_special_tokens) > 0:
        special_splitter = "|".join(esc_special_tokens)
    else:
        special_splitter = None

    # pretokenize in chunks
    PAT = (
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"
        r" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    )
    pretoken_counts: dict[tuple[bytes, ...], int] = {}
    jobs = []
    with open(input_path, "rb") as f:
        num_chunks = 12
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            jobs.append(
                (input_path, start, end, PAT, special_splitter)
            )

    with Pool() as pool:
        pretoken_count_batches = pool.map(pretokenize_chunk, jobs)

    for batch in pretoken_count_batches:
        for ptk, count in batch.items():
            pretoken_counts[ptk] = pretoken_counts.get(ptk, 0) + count
    print(f'\t\tPretokenization complete.')

    # first, count up pairs using base vocab
    merge_hist: list[tuple[bytes, bytes]] = []
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    (pair_counts, pair_to_pretokens) = compute_pair_counts(pretoken_counts)

    # now merge until vocab upper limit or until no more merging possible
    while (len(vocab) < vocab_size) and (len(pair_counts) > 0):
        if len(vocab) % 1000 == 0:
            print(f'\t\t{len(vocab)} tokens encoded.')

        # find max pair
        best_pair, best_count = \
            max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))

        # add new vocab item
        merged_bytes = b''.join(best_pair)
        vocab[len(vocab)] = merged_bytes
        merge_hist.append(best_pair)

        old_set = pair_to_pretokens[best_pair].copy()
        old_counts = {ptk: pretoken_counts[ptk] for ptk in old_set}

        for ptk in old_set:
            count: int = old_counts[ptk]

            # build post-merge pretoken
            new_ptk: list[bytes] = []
            i: int = 0
            is_merge_hit: bool = False
            while i < len(ptk):
                curr_pair = ptk[i:i+2]
                if curr_pair == best_pair:
                    new_ptk.append(merged_bytes)
                    i += 2
                    is_merge_hit = True
                else:
                    new_ptk.append(ptk[i])
                    i += 1
            new_pretoken: tuple[bytes, ...] = tuple(new_ptk)

            # compute old local adjacent pairs
            old_ptk_pairs: set[tuple[bytes, bytes]] = set()
            for i in range(len(ptk)-1):
                pair = ptk[i:i+2]
                pair_counts[pair] -= count
                assert pair_counts[pair] >= 0
                if pair_counts[pair] == 0:
                    pair_counts.pop(pair, None)
                old_ptk_pairs.add(pair)

            for pair in old_ptk_pairs:
                pair_to_pretokens[pair].discard(ptk)
                if len(pair_to_pretokens[pair]) == 0:
                    pair_to_pretokens.pop(pair, None)

            # compute new local adjacent pairs
            for i in range(len(new_pretoken)-1):
                pair = new_pretoken[i:i+2]

                # if pair in pair_counts:
                #     pair_counts[pair] += count
                # else:
                #     pair_counts[pair] = count
                pair_counts[pair] = pair_counts.get(pair, 0) + count

                if pair not in pair_to_pretokens:
                    pair_to_pretokens[pair] = set()
                pair_to_pretokens[pair].add(new_pretoken)

            # update pretoken counts
            pretoken_counts[ptk] -= count
            if pretoken_counts[ptk] == 0:
                pretoken_counts.pop(ptk, None)

            # if new_pretoken in pretoken_counts:
            #     pretoken_counts[new_pretoken] += count
            # else:
            #     pretoken_counts[new_pretoken] = count
            pretoken_counts[new_pretoken] = \
                pretoken_counts.get(new_pretoken,
            0) + count

    return (vocab, merge_hist)


def compute_pair_counts(
    pretoken_counts: dict[tuple[bytes, ...], int]
) -> tuple[
        dict[tuple[bytes, bytes], int],
        dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
    ]:
    """
    helper function for unoptimized merge step
    """

    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_pretokens: \
        dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    for pretoken, count in pretoken_counts.items():
        if len(pretoken) < 2:
            continue

        for i in range(len(pretoken)-1):
            pair = pretoken[i:i+2]
            # if pair in pair_counts:
            #     pair_counts[pair] += count
            # else:
            #     pair_counts[pair] = count
            pair_counts[pair] = pair_counts.get(pair, 0) + count

            if pair in pair_to_pretokens:
                pair_to_pretokens[pair].add(pretoken)
            else:
                ptk_set = set([pretoken])
                pair_to_pretokens[pair] = ptk_set

    return (pair_counts, pair_to_pretokens)

def pretokenize_chunk(
    job_desc
) -> dict[tuple[bytes, ...], int]:
    """
    for parallelizing pretokenization
    """

    (input_path, start, end, regex_pat, special_pat) = job_desc
    pretoken_counts: dict[tuple[bytes, ...], int] = {}
    compiled_pat = re.compile(regex_pat)
    if special_pat:
        compiled_special = re.compile(special_pat)

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        if special_pat:
            split_chunks = compiled_special.split(chunk)
        else:
            split_chunks = [chunk]

        for split in split_chunks:
            match_iter = compiled_pat.finditer(split)
            for match in match_iter:
                word = match.group(0)
                bword = word.encode('utf-8')
                ptk = tuple(bword[i:i+1] for i in range(len(bword)))
                # if ptk in pretoken_counts:
                #     pretoken_counts[ptk] += 1
                # else:
                #     pretoken_counts[ptk] = 1
                pretoken_counts[ptk] = pretoken_counts.get(ptk, 0) + 1

    return pretoken_counts
