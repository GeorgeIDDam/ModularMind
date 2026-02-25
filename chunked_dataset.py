# Dataset loader placeholder

# chunked_dataset.py
import os
import numpy as np
import torch
from torch.utils.data import IterableDataset

class ChunkedTokenDataset(IterableDataset):
    """
    Streaming chunked dataset for language model pretraining.
    - Shuffles chunks every epoch
    - Iterates windows inside each chunk
    - No padding unless needed at chunk tail
    """

    def __init__(
        self,
        chunk_dir: str,
        block_size: int,
        pad_token_id: int,
        shuffle_chunks: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.chunk_dir = chunk_dir
        self.block_size = block_size
        self.pad = pad_token_id
        self.shuffle_chunks = shuffle_chunks
        self.seed = seed

        self.chunk_files = sorted([
            os.path.join(chunk_dir, f)
            for f in os.listdir(chunk_dir)
            if f.endswith(".npy")
        ])

        assert len(self.chunk_files) > 0, "No chunk files found!"

    def set_epoch(self, epoch: int):
        """Called by trainer at start of each epoch"""
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + getattr(self, "epoch", 0))

        chunk_files = self.chunk_files.copy()
        if self.shuffle_chunks:
            rng.shuffle(chunk_files)

        for chunk_path in chunk_files:
            tokens = np.load(chunk_path, mmap_mode="r")

            # slide over chunk
            max_start = len(tokens) - self.block_size - 1
            for start in range(0, max_start, self.block_size):
                window = tokens[start:start + self.block_size + 1]

                if len(window) < self.block_size + 1:
                    pad_len = self.block_size + 1 - len(window)
                    window = np.concatenate(
                        [window, np.full(pad_len, self.pad, dtype=window.dtype)]
                    )

                x = torch.from_numpy(window[:-1]).long()
                y = torch.from_numpy(window[1:]).long()

                yield x, y
