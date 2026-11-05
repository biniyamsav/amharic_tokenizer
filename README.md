# Amharic Tokenizer

A subword (BPE) tokenizer for Amharic (Ge'ez script), trained on about 40M characters of cleaned Amharic text. It includes a from-scratch BPE implementation, a SentencePiece-based production tokenizer, and a benchmark against widely used tokenizers.

## Why

Byte-level tokenizers such as GPT's split Amharic into raw bytes: each Ge'ez character takes 3 bytes in UTF-8, so a typical Amharic word becomes 8-10 tokens. That makes Amharic text expensive and slow for language models. A tokenizer trained on Amharic needs far fewer tokens for the same text.

## Results

Evaluated on 32,417 held-out sentences (lower tokens/word is better):

| Tokenizer | Vocab | Tokens/word | Chars/token | 1-token words | 4+ token words |
|---|---|---|---|---|---|
| **Ours 8k** | 8,000 | 1.65 | 2.79 | 61.4% | 5.9% |
| **Ours 20k** | 20,000 | 1.40 | 3.30 | 74.5% | 2.8% |
| **Ours 32k (default)** | 32,000 | 1.31 | 3.52 | 79.4% | 2.0% |
| **Ours 50k** | 50,000 | 1.25 | 3.69 | 83.0% | 1.5% |
| XLM-R | 250,000 | 2.28 | 2.02 | 41.8% | 20.1% |
| GPT-4o (o200k_base) | 200,019 | 7.93 | 0.58 | 2.9% | 84.5% |
| GPT-4 (cl100k_base) | 100,277 | 10.38 | 0.44 | 2.9% | 84.5% |

The default 32k tokenizer needs about 40% fewer tokens than XLM-R's multilingual tokenizer, with a vocabulary 8x smaller, and 6-8x fewer tokens than GPT's tokenizers. Going from 20k to 32k reduces tokens by 6%, and 32k to 50k by another 5%, while vocabulary usage on unseen text drops from 98% to 95%. 32k is the balance point, and 50k is the larger variant.

## Usage

```python
from src.amharic_tokenizer.sentencepiece_wrapper import AmharicTokenizer

tok = AmharicTokenizer("models/am_bpe_32k.model")

tok.tokenize("ሰላም ነው። እንዴት ነህ?")
# ['▁ሰላም', '▁ነው', '▁።', '▁እንዴት', '▁ነህ', '?']

ids = tok.encode("ሰላም ነው። እንዴት ነህ?")
tok.decode(ids)
```

Text is normalized before tokenizing: duplicate-sound letters are merged (ሠ becomes ሰ, ዐ becomes አ, ሐ/ኀ become ሀ, ፀ becomes ጸ), Ethiopic punctuation is split off into its own tokens, and whitespace is cleaned. Because of this, `decode` returns the normalized text, not the original spelling. Rare characters and emoji fall back to bytes, so nothing becomes an unknown token.

## How it was built

1. `scripts/download_data.py`: downloads Amharic Wikipedia.
2. `scripts/prepare_corpus.py`: filters out non-Amharic lines and markup, splits into sentences, normalizes, removes duplicates, and writes train/validation files (291,759 / 32,417 sentences).
3. `scripts/train_sentencepiece.py`: trains a BPE model (`--vocab-size`, `--model-type`).
4. `scripts/benchmark.py`: compares tokenizers on the validation set.

`src/amharic_tokenizer/bpe.py` is a from-scratch BPE implementation, kept for learning and tested, but too slow for large vocabularies. SentencePiece is used for the released models.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest
```

## Limitations

- Training data: Amharic Wikipedia and [describe your second source here]. Text from other domains (social media, spoken-style Amharic) may tokenize worse.
- The validation set comes from the same sources as the training set, so the scores are better than they would be on a completely new source.
- Normalization is lossy by design (it merges homophone letters).
- Only BPE is released so far. Unigram is supported by the training script but not yet evaluated.

## License

MIT