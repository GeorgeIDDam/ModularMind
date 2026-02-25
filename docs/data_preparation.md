# Data Preparation

## Dataset Structure

### Raw Data
Place your raw data files in `data/raw/`

### Preprocessing
Run preprocessing scripts to prepare data for training.

### Tokenization

#### Training the Tokenizer
```bash
python scripts/train_tokenizer.py
```

#### Tokenizing Data
```bash
python scripts/tokenizer_and_chunk.py
```

The BPE tokenizer configuration is stored in `tokenizer/bpe_tokenizer.json`

## Data Pipeline

1. **Raw Data** → `data/raw/`
2. **Preprocessing** → `data/preprocessed/`
3. **Tokenization** → `data/tokenized/`
4. **Training** → Uses tokenized data

## Dataset Classes

### `dataset.py`
[To be documented: Describe the dataset class]

### `chunked_dataset.py`
[To be documented: Describe chunked dataset implementation]

## Data Format Requirements
[To be documented: Specify expected data formats]
