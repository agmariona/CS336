import numpy as np
from pathlib import Path

tokens = np.load("data/ts_train_tokens.npy", mmap_mode='r')

n_tokens = 2048
debug_tokens = np.array(tokens[:n_tokens], dtype=tokens.dtype)
np.save("data/ts_debug_tokens.npy", debug_tokens)
