from cs336_basics.bpe_trainer import bpe_trainer
import json

def main():
    datapath = 'data/owt_train.txt'
    vocab_size = 32000
    print(f"\tTraining BPE tokenizer on {datapath}",
        f"with a vocab size of {vocab_size}.")
    vocab, merges = bpe_trainer(datapath, vocab_size, ['<|endoftext|>'])

    serial_vocab = []
    for tid, tbytes in vocab.items():
        serial_vocab.append({
            "id": tid,
            "bytes_hex": tbytes.hex()
        })
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(serial_vocab, f, indent=2)

    serial_merges = []
    for pair in merges:
        serial_merges.append({
            "left_byte_hex": pair[0].hex(),
            "right_byte_hex": pair[1].hex()
        })
    with open('merges.json', 'w', encoding='utf-8') as f:
        json.dump(serial_merges, f, indent=2)

if __name__ == '__main__':
    main()
