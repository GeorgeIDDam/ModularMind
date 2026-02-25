# python analyze_config.py
# This script analyzes the model configuration from train.yaml and model.py, estimates the number of parameters, memory requirements, and provides recommendations for training on an RTX 3060
# Usage: python analyze_config.py

import torch
from model import GPTConfig

# Model from config.yaml
config_yaml = GPTConfig(
    vocab_size=20000,
    block_size=1024,
    n_layer=6,
    n_head=8,
    n_embd=512,
    dropout=0.1
)

# Model from model.py  
config_py = GPTConfig(
    vocab_size=20000,
    block_size=2048,
    n_layer=8,
    n_head=12,
    n_embd=768,
    dropout=0.1
)

def calculate_params(config):
    vocab_size = config.vocab_size
    n_embd = config.n_embd
    n_layer = config.n_layer
    n_head = config.n_head
    n_inner = getattr(config, 'n_inner', 4 * n_embd)
    
    # Token + position embeddings
    embed_params = vocab_size * n_embd + config.block_size * n_embd
    
    # Each transformer block
    # Attention: 3 * n_embd^2 (Q,K,V) + n_embd^2 (proj)
    attn_params = 4 * n_embd * n_embd
    
    # MLP: n_embd * n_inner + n_inner * n_embd
    mlp_params = 2 * n_embd * n_inner
    
    # Layer norms: 2 * n_embd per block  
    ln_params = 2 * n_embd
    
    block_params = (attn_params + mlp_params + ln_params) * n_layer
    
    # Final layer norm + output head
    final_params = n_embd + vocab_size * n_embd
    
    total = embed_params + block_params + final_params
    return total, embed_params, block_params, final_params

def estimate_memory(params, batch_size, seq_len, dtype_bytes=2):
    # Model parameters (2 bytes for fp16)
    model_memory = params * dtype_bytes
    
    # Gradients (same size as parameters)  
    grad_memory = params * dtype_bytes
    
    # Optimizer states (AdamW: 2x params for momentum + variance)
    optimizer_memory = params * dtype_bytes * 2
    
    # Activations (rough estimate)
    activation_memory = batch_size * seq_len * 512 * dtype_bytes * 6  # layers
    
    total_gb = (model_memory + grad_memory + optimizer_memory + activation_memory) / 1e9
    return total_gb, model_memory/1e9, grad_memory/1e9, optimizer_memory/1e9, activation_memory/1e9

print("=== CONFIGURATION ANALYSIS FOR RTX 3060 12GB ===\n")

# Calculate parameters
params_yaml, embed_yaml, block_yaml, final_yaml = calculate_params(config_yaml)
params_py, embed_py, block_py, final_py = calculate_params(config_py)

print(f"YAML Config (train.yaml):  {params_yaml/1e6:.1f}M parameters")
print(f"Python Config (model.py):  {params_py/1e6:.1f}M parameters\n")

# Memory analysis
batch_size = 16
seq_len = 1024

memory_yaml, model_mem_yaml, grad_mem_yaml, opt_mem_yaml, act_mem_yaml = estimate_memory(params_yaml, batch_size, seq_len)
memory_py, model_mem_py, grad_mem_py, opt_mem_py, act_mem_py = estimate_memory(params_py, batch_size, seq_len)

print(f"Memory Requirements (batch_size={batch_size}, seq_len={seq_len}):")
print(f"YAML config:    {memory_yaml:.1f}GB")
print(f"  - Model:      {model_mem_yaml:.1f}GB")
print(f"  - Gradients:  {grad_mem_yaml:.1f}GB")  
print(f"  - Optimizer:  {opt_mem_yaml:.1f}GB")
print(f"  - Activations:{act_mem_yaml:.1f}GB")
print()
print(f"Python config:  {memory_py:.1f}GB")
print(f"  - Model:      {model_mem_py:.1f}GB")
print(f"  - Gradients:  {grad_mem_py:.1f}GB")
print(f"  - Optimizer:  {opt_mem_py:.1f}GB")
print(f"  - Activations:{act_mem_py:.1f}GB")
print()

print("GPU: RTX 3060 12GB")
print(f"Available memory: ~11.5GB (after OS overhead)")
print()

# Recommendations
print("=== RECOMMENDATIONS ===")
if memory_yaml <= 11.5:
    print("✅ YAML config fits in GPU memory")
else:
    print("❌ YAML config may cause OOM")
    
if memory_py <= 11.5:
    print("✅ Python config fits in GPU memory")
else:
    print("❌ Python config may cause OOM")

print()
print("Optimal batch sizes:")
for bs in [8, 12, 16, 20, 24]:
    mem_test = estimate_memory(params_yaml, bs, seq_len)[0]
    status = "✅" if mem_test <= 11.5 else "❌"
    print(f"  Batch size {bs:2d}: {mem_test:.1f}GB {status}")