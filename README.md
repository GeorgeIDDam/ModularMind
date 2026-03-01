# ModularMind - Machine Learning Model Project

## Thesis
ModularMind explores whether structured modular specialization and orchestration can increase reasoning depth without requiring massive scale.

## Overview
ModularMind is an experimental Transformer-based AI framework built from scratch on consumer-grade hardware.

The project investigates:

- Custom BPE tokenizer training

- Streaming dataset pipelines

- Stable Transformer training under hardware constraints

- EMA stabilization strategies

- Learning rate control and debugging

- Modular checkpoint specialization

- Multi-model orchestration (future phase)

This is not an API wrapper.
This is a ground-up architecture experiment.

## Baseline Architecture(v0.1 Stable)
Model type: Decoder-only Transformer(GPT-style)

| Parameter           | Value   |
| ------------------- | ------- |
| vocab_size          | 32000   |
| block_size          | 2048    |
| n_layer             | 10      |
| n_head              | 12      |
| n_embd              | 768     |
| head_dim            | 64      |
| dropout             | 0.1     |
| use_rmsnorm         | True    |
| use_rope            | True    |
| use_flash_attention | False   |
| rope_theta          | 10000.0 |


## Training Baseline
- Dataset: Cleaned streaming chunk dataset

- Tokenizer: Custom BPE (32k vocab)

- Batch size: 2

- Learning rate: 3e-5 (stable phase)

- Gradient clipping: 0.5

- EMA: Enabled

- Mixed precision (AMP): Enabled

- Best validation loss: ~3.54

- Hardware: RTX 3060 / RTX 5060

## Project Structure
```
ModularMind/
├── model.py
├── trainer.py
├── train.py
├── chunked_dataset.py
├── generate.py
├── autopilot.py
├── model_config.py
├── config.py
├── scripts/
│   ├── train_tokenizer.py
│   └── tokenizer_and_chunk.py
├── tokenizer/
├── data/
├── configs/
└── logs/
```
### Key Engineering Milestones

- Fixed NaN instabilities

- Corrected EMA + AMP interaction

- Removed unintended LR override during validation

- Replaced in-memory dataset with streaming IterableDataset

- Stabilized convergence under constrained hardware

- Achieved consistent best checkpoint tracking


### Roadmap
- Fixed NaN instabilities

- Corrected EMA + AMP interaction

- Removed unintended LR override during validation

- Replaced in-memory dataset with streaming IterableDataset

- Stabilized convergence under constrained hardware

- Achieved consistent best checkpoint tracking

## Vision

Rather than relying solely on scale, ModularMind explores whether modular specialization, structured reasoning loops, and orchestration strategies can produce deeper reasoning behavior within limited compute environments.

Guiding question:

  Can structured modularity compete with brute-force scale?

## Credits

  Jorge I. Dávila Mtz.


