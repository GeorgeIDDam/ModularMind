import numpy as np
import torch
from torch.utils.data import Dataset

class ChunkedTokenDataset(Dataset):
    def __init__(self, bin_path, block_size, start_block=0, end_block=None):
        self.block_size = block_size
        self.tokens = np.memmap(bin_path, dtype=np.int32, mode="r")

        total_blocks = (len(self.tokens) - 1) // block_size
        self.start = start_block
        self.end = end_block or total_blocks
        assert self.start < self.end

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        idx = self.start + idx
        start = idx * self.block_size
        end = start + self.block_size + 1
        window = np.array(self.tokens[start:end], copy=True)

        x = torch.from_numpy(window[:-1]).long()
        y = torch.from_numpy(window[1:]).long()
        return x, y
