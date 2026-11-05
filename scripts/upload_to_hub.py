"""Upload the trained tokenizers and a model card to the Hugging Face Hub."""
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "biniyam882/amharic-bpe-tokenizer"

MODEL_FILES = [
    "models/am_bpe_32k.model",
    "models/am_bpe_32k.vocab",
    "models/am_bpe_50k.model",
    "models/am_bpe_50k.vocab",
    "models/am_bpe_20k.model",
    "models/am_bpe_20k.vocab",
]

MODEL_CARD = """---
language:
- am
license: mit
tags:
- tokenizer
- bpe
- sentencepiece
- amharic
---

# Amharic BPE Tokenizer

SentencePiece BPE tokenizers for Amharic (Ge'ez script), in three vocabulary sizes.
Code, training pipeline and benchmark: https://github.com/biniyamsav/amharic_tokenizer

| File | Vocab | Tokens/word | Chars/token |
|---|---|---|---|
| `am_bpe_32k.model` (default) | 32,000 | 1.31 | 3.52 |
| `am_bpe_50k.model` | 50,000 | 1.25 | 3.69 |
| `am_bpe_20k.model` | 20,000 | 1.40 | 3.30 |

For comparison on the same held-out text: XLM-R 2.28, GPT-4o 7.93, GPT-4 10.38 tokens per word.

## Important: normalize text first

These models were trained on normalized text (duplicate-sound letters merged,
Ethiopic punctuation separated by spaces). Raw text should go through
`normalize_text` from the GitHub repo before encoding. The repo's
`AmharicTokenizer` class does this for you.

```python
import sentencepiece as spm
from huggingface_hub import hf_hub_download

path = hf_hub_download("biniyam882/amharic-bpe-tokenizer", "am_bpe_32k.model")
sp = spm.SentencePieceProcessor(model_file=path)
print(sp.encode("ሰላም ነው ። እንዴት ነህ ?", out_type=str))
```

## Limitations

Trained on Amharic Wikipedia and one additional web corpus (about 40M characters).
Other domains may tokenize worse. Normalization is lossy by design.
"""


def main():
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
    )

    for file_path in MODEL_FILES:
        path = Path(file_path)
        print(f"Uploading {path.name}...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=REPO_ID,
            repo_type="model",
        )

    print(f"Done: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()