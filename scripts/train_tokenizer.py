# RUN THIS FIRST: python scripts/train_tokenizer.py

#train tokenizer vocab size same as model:# RUN THIS FIRST: python scripts/train_tokenizer.py

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from pathlib import Path
import json
import hashlib

# ======================
# CONFIG
# ======================
CORPUS_PATH = "data/raw/corpus.txt"
TOKENIZER_OUT = "tokenizer/bpe_tokenizer_v2.json"

VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]

# ======================
# TRAIN TOKENIZER
# ======================
def main():
    print("🚀 Training BPE tokenizer from scratch")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train(files=[CORPUS_PATH], trainer=trainer)

    tokenizer.save(TOKENIZER_OUT)
    print(tokenizer.to_str())


    vocab = tokenizer.get_vocab()

    assert vocab["[PAD]"] == 0, "PAD token must be id 0"
    assert vocab["[BOS]"] == 1
    assert vocab["[EOS]"] == 2
    assert vocab["[UNK]"] == 3


    Path("tokenizer").mkdir(parents=True, exist_ok=True)
    tokenizer.save(TOKENIZER_OUT)


    ### Save tokenizer config for training
    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()


    tokenizer_config = {
        "type": "ByteLevel BPE",
        "vocab_size": VOCAB_SIZE,
        "special_tokens": {
            tok: tokenizer.get_vocab()[tok]
            for tok in SPECIAL_TOKENS
        },
        "normalizer": "NFKC",
        "pre_tokenizer": "ByteLevel(add_prefix_space=True)",
        "decoder": "ByteLevel",
        "corpus_path": CORPUS_PATH,
        "tokenizer_file": TOKENIZER_OUT,
        "tokenizer_sha256": sha256(TOKENIZER_OUT),
    }

    with open("tokenizer/tokenizer_config_v2.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)

    print("🧾 Tokenizer config saved to tokenizer/tokenizer_config_v2.json")
    ### save tokenizer config for training, including vocab size and special token ids, so we can verify it in train.py

    vocab = tokenizer.get_vocab()
    print(f"✅ Tokenizer trained")
    print(f"   Vocab size: {len(vocab)}")
    print("   Special tokens:")
    for tok in SPECIAL_TOKENS:
        print(f"     {tok}: {vocab[tok]}")

if __name__ == "__main__":
    main()
