# trainer_usage.py
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from model import TransformerLM
from chunked_dataset import ChunkedTokenDataset
from train import ModelConfig
# =========================
dataset = ChunkedTokenDataset(
    chunk_dir="data/preprocessed",
    block_size=1024,
    pad_token_id=0,
    shuffle_chunks=True,
    seed=42,
)

train_loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)

model = TransformerLM(ModelConfig()).to("cuda" if torch.cuda.is_available() else "cpu")

optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.1,
)   
# =========================
# TRAINER USAGE EXAMPLE and SETUP ABOVE ADDED BY USER
# =========================
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    pad_token_id=0,
    grad_clip=1.0,
    gradient_accumulation_steps=8,
    log_interval=50,
)

trainer.train(num_epochs=200)
