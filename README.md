# ModularMind - Machine Learning Model Project

## Thesis
ModularMind explores whether structured modular specialization and orchestration can increase reasoning depth without requiring massive scale.

## Overview
This project implements a machine learning model with custom training pipeline and tokenization.

## Project Structure
```
JIDM_v2/
├── model.py              # Model architecture definition
├── trainer.py            # Training loop and logic
├── train.py              # Training script entry point
├── dataset.py            # Dataset handling
├── chunked_dataset.py    # Chunked dataset implementation
├── config.py             # Configuration management
├── model_config.py       # Model-specific configuration
├── generate.py           # Text generation
├── autopilot.py          # Automated training
├── configs/              # Configuration files
│   └── train.yaml        # Training configuration
├── data/                 # Data directory
│   ├── raw/              # Raw data files
│   ├── preprocessed/     # Preprocessed data
│   ├── tokenized/        # Tokenized data
│   ├── checpoints/       # Model checkpoints
│   └── logs/             # Training logs
├── scripts/              # Utility scripts
│   ├── train_tokenizer.py
│   └── tokenizer_and_chunk.py
├── tokenizer/            # Tokenizer files
│   └── bpe_tokenizer.json
└── utils/                # Utility functions
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py
```

### Generation
```bash
python generate.py
```

## Configuration
Edit `configs/train.yaml` to modify training parameters.

## License
Private project - All rights reserved

## Contributors
[Your Name]
