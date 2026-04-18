from bpe_tokenizer import bpe_tokenizer

datapath = '../data/small.txt'
pretoken_counts, merges = bpe_tokenizer(datapath, 1024, ['<|endoftext|>'])

for k,v in pretoken_counts.items():
    print(k, ':', v)
print('\n')
for m in merges:
    print(m)

# print('\n\n ---------- \n\n')
#
# for key, value in freqs.items():
#     print(key, ':', value)
#
# print('\n')
#
# for key, value in p_index.items():
#     print(key, ':', value)
