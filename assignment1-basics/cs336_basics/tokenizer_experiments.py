from cs336_basics.tokenizer import Tokenizer
import numpy as np
import time
import random

def main():
    ts_data_path = 'data/TinyStoriesV2-GPT4-train.txt'
    ts_valid_path = 'data/TinyStoriesV2-GPT4-valid.txt'
    ts_vocab_path = 'data/tinystories-vocab.json'
    ts_merges_path = 'data/tinystories-merges.json'

    owt_data_path = 'data/owt_train.txt'
    owt_valid_path = 'data/owt_valid.txt'
    owt_vocab_path = 'data/owt-vocab.json'
    owt_merges_path = 'data/owt-merges.json'

    special_tokens = ['<|endoftext|>']

    ts_tokenizer = Tokenizer.from_files(
        ts_vocab_path, ts_merges_path, special_tokens)
    owt_tokenizer = Tokenizer.from_files(
        owt_vocab_path, owt_merges_path, special_tokens)

    # encode corpus

    # with open(ts_data_path, "r", encoding="utf-8") as f:
    #     ts_train_enc_iter = ts_tokenizer.encode_iterable(f)
    #     ts_train_tokens = np.fromiter(ts_train_enc_iter, dtype=np.uint16)
    # np.save('data/ts_train_tokens.npy', ts_train_tokens)

    # with open(ts_valid_path, "r", encoding="utf-8") as f:
    #     ts_valid_enc_iter = ts_tokenizer.encode_iterable(f)
    #     ts_valid_tokens = np.fromiter(ts_valid_enc_iter, dtype=np.uint16)
    # np.save('data/ts_valid_tokens.npy', ts_valid_tokens)

    with open(owt_data_path, "r", encoding="utf-8") as f:
        owt_train_enc_iter = owt_tokenizer.encode_iterable(f)
        owt_train_tokens = np.fromiter(owt_train_enc_iter, dtype=np.uint16)
    np.save('data/owt_train_tokens.npy', owt_train_tokens)

    with open(owt_valid_path, "r", encoding="utf-8") as f:
        owt_valid_enc_iter = owt_tokenizer.encode_iterable(f)
        owt_valid_tokens = np.fromiter(owt_valid_enc_iter, dtype=np.uint16)
    np.save('data/owt_valid_tokens.npy', owt_valid_tokens)

    return

    # measure throughput
    documents, total_bytes = partial_load(owt_data_path, int(250e6))
    n_trials = 3
    t_bulk = 0
    for i in range(n_trials):
        t_start = time.perf_counter()
        tokens = 0
        for doc in documents:
            tokens += len(owt_tokenizer.encode(doc))
        t_end = time.perf_counter()
        t_bulk += t_end - t_start
    t_avg = t_bulk / n_trials
    print(f'Average throughput: {total_bytes / t_avg} bytes / second')
    print(f'                    {tokens / t_avg} tokens / second')

    return

    # sample 10 stories and compute compression ratio
    ts_sample = sample_documents(ts_data_path)
    owt_sample = sample_documents(owt_data_path)

    total_bytes = 0
    total_tokens = 0
    for sample in owt_sample:
        sample_enc = ts_tokenizer.encode(sample)
        total_bytes += len(sample.encode('utf-8'))
        total_tokens += len(sample_enc)
    ts_compression_ratio = total_bytes / total_tokens
    print(f'TinyStories empirical compression ratio: {ts_compression_ratio}')

    total_bytes = 0
    total_tokens = 0
    for sample in ts_sample:
        sample_enc = owt_tokenizer.encode(sample)
        total_bytes += len(sample.encode('utf-8'))
        total_tokens += len(sample_enc)
    owt_compression_ratio = total_bytes / total_tokens
    print(f'OpenWebText empirical compression ratio: {owt_compression_ratio}')


def partial_load(
    path: str,
    target_size: int,
    delim: str = "<|endoftext|>"
) -> tuple[list[str], int]:
    documents = []
    buffer = ""
    total_bytes = 0

    with open(path, "r", encoding="utf-8") as f:
        for chunk in iter(lambda: f.read(1 << 20), ""):  # 1 MB chunks
            buffer += chunk
            parts = buffer.split(delim)

            # all complete docs except the last remainder
            for doc in parts[:-1]:
                if not doc:
                    continue
                documents.append(doc)
                total_bytes += len(doc.encode('utf-8'))

                if total_bytes > target_size:
                    return (documents, total_bytes)

            buffer = parts[-1]

    return documents, total_bytes

def sample_documents(
    path: str,
    k: int = 10,
    delim: str = "<|endoftext|>"
) -> list[str]:
    reservoir = []
    seen = 0
    buffer = ""

    with open(path, "r", encoding="utf-8") as f:
        for chunk in iter(lambda: f.read(1 << 20), ""):  # 1 MB chunks
            buffer += chunk
            parts = buffer.split(delim)

            # all complete docs except the last remainder
            for doc in parts[:-1]:
                if not doc:
                    continue

                seen += 1
                if len(reservoir) < k:
                    reservoir.append(doc)
                else:
                    j = random.randrange(seen)
                    if j < k:
                        reservoir[j] = doc

            buffer = parts[-1]

    # optional: handle trailing doc if file does not end with delimiter
    if buffer:
        seen += 1
        if len(reservoir) < k:
            reservoir.append(buffer)
        else:
            j = random.randrange(seen)
            if j < k:
                reservoir[j] = buffer

    return reservoir

if __name__ == '__main__':
    main()
