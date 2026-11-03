# amharic-tokenizer

A tokenizer for Amharic (Ge'ez/Ethiopic script).

## Motivation

Byte-level tokenizers fragment Ge'ez script characters, which are encoded as
roughly 3 bytes per character in UTF-8. This project explores dedicated
tokenizer approaches for Amharic.

## Planned features

- From-scratch BPE implementation
- SentencePiece (BPE and Unigram) baselines
- Evaluation suite: tokens per word, compression ratio, vocab coverage,
  comparison against a byte-level tokenizer on the same text

## Project structure

```
amharic-tokenizer/
├── src/amharic_tokenizer/   # library code
├── scripts/                 # training / evaluation CLIs
├── tests/                   # pytest tests
├── data/                    # raw and processed corpus (gitignored contents)
├── models/                  # trained models (gitignored)
├── results/                 # evaluation outputs
└── docs/                    # notes
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Status: in development

## Results

| Tokenizer | Tokens/word | Compression | Vocab coverage |
|-----------|-------------|-------------|----------------|
| BPE (ours)| TBD         | TBD         | TBD            |
| SentencePiece BPE | TBD | TBD         | TBD            |
| SentencePiece Unigram | TBD | TBD     | TBD            |
| Byte-level | TBD         | TBD         | TBD            |
