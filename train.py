# train.py
# THRID RUN THIS: python train.py

import torch
import yaml
from dataset import ChunkedTokenDataset
from trainer import Trainer
from model import TransformerLM
from config import TransformerConfig

import json
from tokenizers import Tokenizer

with open("tokenizer/tokenizer_config.json") as f:
    cfg = json.load(f)

tokenizer = Tokenizer.from_file(cfg["tokenizer_file"])

assert tokenizer.get_vocab_size() == cfg["vocab_size"]
assert tokenizer.get_vocab()["[PAD]"] == cfg["special_tokens"]["[PAD]"]

print("✅ Tokenizer config verified")



BIN = "data/tokenized/train.bin"
BLOCK = 1024
VOCAB = 20_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ChunkedTokenDataset(BIN, BLOCK)
n = len(dataset)
train_ds = ChunkedTokenDataset(BIN, BLOCK, 0, int(n * 0.99))
val_ds = ChunkedTokenDataset(BIN, BLOCK, int(n * 0.99), n)
vocab = tokenizer.get_vocab()

assert vocab["[PAD]"] == 0, "PAD token must be id 0"
assert vocab["[BOS]"] == 1
assert vocab["[EOS]"] == 2
assert vocab["[UNK]"] == 3

model = TransformerLM(TransformerConfig( 
    vocab_size=cfg["vocab_size"],
    n_positions=1024,
    n_embd=512,
    n_head=8,
    n_layer=6,
    n_inner=2048,
    pre_norm=True,
    norm_type="rmsnorm",
    rotary_emb_base=10000,
    flash_attention=True
))

trainer = Trainer(
    model, train_ds, val_ds,
    batch_size=16,
    lr=1e-4,
    warmup_steps=750,
    max_steps=200_000,
    grad_clip=1.0,
    device=DEVICE,
    min_frequency=1,
)

for epoch in range(1, 201):
    print(f"\n🌍 Epoch {epoch}")
    trainer.train_epoch(epoch)
    val_loss = trainer.validate()
    trainer.save(epoch, val_loss)