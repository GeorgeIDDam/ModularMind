#TO RUN THIS: python generate.py

import torch
from tokenizers import Tokenizer
from utils.model import TransformerLM
from config import TransformerConfig

tokenizer = Tokenizer.from_file("tokenizer/bpe_tokenizer.json")
model = TransformerLM(TransformerConfig(  
    vocab_size=20000,
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
model.load_state_dict(torch.load("checkpoints/ckpt_200.pt")["model"])
model.eval().cuda()

prompt = "Once upon a time"
ids = tokenizer.encode(prompt).ids
x = torch.tensor(ids)[None].cuda()

for _ in range(200):
    logits, _ = model(x)
    next_id = torch.argmax(logits[:, -1], dim=-1)
    x = torch.cat([x, next_id[:, None]], dim=1)

print(tokenizer.decode(x[0].tolist()))
