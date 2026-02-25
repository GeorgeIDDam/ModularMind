# RUN THIS FIRST: python scripts/tokenizer_and_chunk.py
# TOKENIZE THE CORPUS AND CHUNK INTO FIXED-SIZE SEQUENCES

import os
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

# ======================
# CONFIG
# ======================
CORPUS_PATH = "data/raw/corpus.txt"
TOKENIZER_CONFIG_PATH = "tokenizer/tokenizer_config_v2.json"
OUT_DIR = "data/preprocessed_v2"

BLOCK_SIZE = 4096
STRIDE = BLOCK_SIZE // 4
CHUNKS_PER_FILE = 1024
PAD_TOKEN_ID = 0

# ======================
# MAIN
# ======================
def main():
    print("📚 Loading tokenizer config")

    with open(TOKENIZER_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = Tokenizer.from_file(cfg["tokenizer_file"])
    vocab_size = tokenizer.get_vocab_size()

    assert vocab_size == cfg["vocab_size"]
    assert tokenizer.get_vocab()["[PAD]"] == cfg["special_tokens"]["[PAD]"]

    print("✅ Tokenizer config verified")
    print(f"   Vocab size: {vocab_size}")
    print(f"   PAD token id: {PAD_TOKEN_ID}")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    token_buffer = []
    xs_buffer = []
    ys_buffer = []
    chunk_idx = 0

    print("📖 Streaming corpus + tokenizing")

    with open(CORPUS_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Tokenizing"):
            line = line.strip()
            if not line:
                continue

            ids = tokenizer.encode(line).ids

            for t in ids:
                if t < 0 or t >= vocab_size:
                    raise ValueError(f"Token id out of range: {t}")

            token_buffer.extend(ids)

            while len(token_buffer) >= BLOCK_SIZE + 1:
                x = token_buffer[:BLOCK_SIZE]
                y = token_buffer[1:BLOCK_SIZE + 1]
                token_buffer = token_buffer[STRIDE:]

                xs_buffer.append(torch.tensor(x, dtype=torch.long))
                ys_buffer.append(torch.tensor(y, dtype=torch.long))

                if len(xs_buffer) >= CHUNKS_PER_FILE:
                    save_chunk(xs_buffer, ys_buffer, chunk_idx)
                    xs_buffer.clear()
                    ys_buffer.clear()
                    chunk_idx += 1

    # Flush leftovers
    if xs_buffer:
        save_chunk(xs_buffer, ys_buffer, chunk_idx)

    print("✅ Tokenization + chunking complete")


def save_chunk(xs, ys, idx):
    xs = torch.stack(xs)
    ys = torch.stack(ys)

    out_path = os.path.join(
        OUT_DIR, f"chunk_{idx:06d}.pt"
    )
    torch.save({"x": xs, "y": ys}, out_path)
    print(f"💾 Saved {out_path}")


if __name__ == "__main__":
    main()
