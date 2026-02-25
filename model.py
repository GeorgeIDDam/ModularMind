# src/model.py
import math
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 20000 # Set this to your tokenizer's vocab size
    block_size: int = 2048  # Increased context length
    n_layer: int = 8        # More layers for better performance
    n_head: int = 12        # More attention heads
    n_embd: int = 768       # Larger embedding dimension
    dropout: float = 0.1
    use_rmsnorm: bool = True
    use_rope: bool = True
    use_flash_attention: bool = True
    rope_theta: float = 10000.0  # RoPE base frequency
    device: Optional[torch.device] = None  # Device can be set externally
    max_seq_len: int = 2048  # Maximum sequence length for positional embeddings (if not using RoPE), should match block_size   
    apply_rotary_pos_emb: bool = True  # Whether to apply RoPE in attention, set to False if using absolute positional embeddings
    
    # Training hyperparameters (set these in trainer as well) From here you can set the training hyperparameters, but they should also be set in the trainer for consistency. You can choose to keep them here for easy access or move them entirely to the trainer.
    gradient_accumulation_steps: int = 1  # For effective larger batch sizes *
    init_lr: float = 1e-4 # Initial learning rate for the optimizer, set this in trainer as well*
    warmup_steps: int = 1000 # Number of warmup steps for learning rate scheduling, set this in trainer as well*S
    max_lr: float = 1e-3 # Peak learning rate for cosine schedule, set this in trainer as well*
    weight_decay: float = 0.01 # L2 regularization for better generalization*
    max_steps: int = 50000 # Total training steps, set this in trainer as well*
    max_steps_without_improvement: int = 25000# Early stopping if no improvement in validation loss for this many steps*
    early_stopping: bool = False #*
    quit_on_oom: bool = True #*
    rms_norm_eps: float = 1e-6  # Epsilon for RMSNorm stability*
    optimizer: str = "AdamW"  # Optimizer choice*
    checkpoint_path: Optional[str] = "data/checkpoints"  # Directory for checkpoints*
    tensorboard_log_dir: Optional[str] = "data/logs"  # Directory for TensorBoard logs*
    use_mixed_precision: bool = True  # Use mixed precision training for efficiency*S
    verbose: bool = True  # Verbose logging during training*
    validation_interval: int = 500  # Validate every N steps*
    validation_samples: int = 500  # Number of samples to use for validation, set in trainer as well*
    save_interval: int = 1000  # Save checkpoint every N steps, set in trainer as well*
    grad_clip: float = 1.0  # Gradient clipping value for stability, set this in trainer as well*
    rnd_seed: int = 42  # Random seed for reproducibility, set this in trainer as well*

# Norma RMS (Root Mean Square Layer Normalization)
class RMSNorm(nn.Module):
    """RMSNorm implementation - more stable than LayerNorm"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation"""
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for positional encodings
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = max(seq_len, self._seq_len_cached)
            
            t = torch.arange(self._seq_len_cached, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        self._update_cache(seq_len, x.device, x.dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Módulos del modelo GPT basado en el paper original "Attention is All You Need" y "Improving Language Understanding by Generative Pre-Training"
# Implementación simplificada para fines educativos y de entrenamiento rápido   
# Puedes ajustar la arquitectura según tus necesidades agregando más capas, atención, etc.
# Uso:
# config = GPTConfig(vocab_size=3000, block_size=512, n_layer
# model = GPT(config)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size
        self.use_rope = config.use_rope
        self.use_flash_attention = config.use_flash_attention

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE for positional encoding
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                max_seq_len=config.block_size,
                theta=config.rope_theta
            )
        else:
            # Fallback to causal mask for non-RoPE
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        
        # Check for flash attention availability
        self.flash = hasattr(F, "scaled_dot_product_attention") and self.use_flash_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # (B, nh, T, hd)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hd)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hd)

        # Apply RoPE if enabled
        if self.use_rope:
            cos, sin = self.rotary_emb(x)
            # Expand cos/sin to match the shape (B, nh, T, hd)
            cos = cos[None, None, :, :].expand(B, self.n_head, T, self.head_dim)
            sin = sin[None, None, :, :].expand(B, self.n_head, T, self.head_dim)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Use flash attention if available and enabled
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0)
        else:
            # Manual attention computation
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if not self.use_rope:  # Only apply mask if not using RoPE
                mask = torch.tril(torch.ones(T, T, device=x.device))
                att = att.masked_fill(mask == 0, float("-inf"))
            else:
                # For RoPE, we still need causal masking in attention
                causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                att = att.masked_fill(~causal_mask, float("-inf"))
            
            att = F.softmax(att, dim=-1)
            y = att @ v
            y = self.attn_dropout(y)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y

# Feed-Forward Network (MLP) dentro del bloque Transformer  
# Puedes ajustar el tamaño de las capas según tus necesidades Agrega más capas si lo deseas 
# Uso interno en el bloque Transformer,, MLP MEANS Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# Bloque Transformer completo: atención + MLP con normalización y conexiones residuales
# Puedes ajustar la arquitectura según tus necesidades agregando más capas, atención, etc.
# Uso interno en el modelo GPT
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use RMSNorm if enabled, otherwise LayerNorm
        norm_class = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.ln1 = norm_class(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = norm_class(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# Modelo GPT completo 
# Puedes ajustar la arquitectura según tus necesidades agregando más capas, atención, etc.
# Uso: 
# config = GPTConfig(vocab_size=3000, block_size=512, n_layer=12, n_head=12, n_embd=768)
class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Only use positional embeddings if RoPE is disabled
        if not config.use_rope:
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        else:
            self.pos_emb = None
        
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Use RMSNorm for final layer norm if enabled
        norm_class = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.ln_f = norm_class(config.n_embd)
        
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, pad_token_id: int = 0) -> Any:
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Secuencia demasiado larga: {T} > {self.config.block_size}")

        device = idx.device
        tok_emb = self.tok_emb(idx)  # (B, T, C)
        
        # Add positional embeddings only if RoPE is disabled
        if self.pos_emb is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
            pos_emb = self.pos_emb(pos)  # (1, T, C)
            x = self.drop(tok_emb + pos_emb)
        else:
            # With RoPE, we only use token embeddings + dropout
            x = self.drop(tok_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=pad_token_id
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def to_config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)

    @staticmethod
    def from_config_dict(cfg: Dict[str, Any]) -> "TransformerLM":
        config = config(**cfg)
        return TransformerLM(config)
