"""Clean raw Amharic corpora and create train/validation files."""

from pathlib import Path
import random
import re
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.amharic_tokenizer.normalize import normalize_text
from src.amharic_tokenizer.utils import (
    ethiopic_char_ratio,
    read_corpus,
    split_train_val,
)

RAW_FILES = [
    "data/raw/amwiki.txt",
    "data/raw/amharic_pretraining.txt",
]

MIN_RATIO = 0.9        # at least 90% of the letters must be Ethiopic
MIN_CHARS = 10         # drop very short sentences
MIN_WORDS = 3          # drop headings and fragments
MAX_CHARS = 1000       # drop absurdly long sentences (SentencePiece skips ~4000 bytes)
MAX_LINES_PER_SOURCE = None   # e.g. 300000 to stop one big file dominating

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TAG_RE = re.compile(r"<[^>]+>")
WIKI_RE = re.compile(r"={2,}|\{\{|\}\}|\[\[|\]\]|\|")   # wiki markup leftovers
SENT_SPLIT_RE = re.compile(r"(?<=[።?!])\s+")
DIGITS_RE = re.compile(r"[0-9]+")

# Everything NOT in this list gets removed: Ethiopic block, digits,
# whitespace, and a few common punctuation marks.
DISALLOWED_RE = re.compile(r"[^\u1200-\u137F0-9\s.,!?:;()\"'%/\-]")


def clean_sentence(s: str):
    """Return a cleaned sentence, or None if it should be dropped."""
    if WIKI_RE.search(s):
        return None
    s = URL_RE.sub(" ", s)
    s = TAG_RE.sub(" ", s)

    if ethiopic_char_ratio(s) < MIN_RATIO:
        return None

    s = DISALLOWED_RE.sub(" ", s)
    s = normalize_text(s)

    if len(s) < MIN_CHARS or len(s) > MAX_CHARS or len(s.split()) < MIN_WORDS:
        return None
    return s


def main():
    seen = set()
    cleaned = []
    stats = {"bad": 0, "dup": 0}

    for file_path in RAW_FILES:
        lines = read_corpus(file_path)
        kept = []

        for line in lines:
            for sentence in SENT_SPLIT_RE.split(line):
                result = clean_sentence(sentence)
                if result is None:
                    stats["bad"] += 1
                    continue
                # near-duplicate key: same sentence with different numbers
                key = DIGITS_RE.sub("0", result)
                if key in seen:
                    stats["dup"] += 1
                    continue
                seen.add(key)
                kept.append(result)

        if MAX_LINES_PER_SOURCE and len(kept) > MAX_LINES_PER_SOURCE:
            kept = random.Random(42).sample(kept, MAX_LINES_PER_SOURCE)

        print(f"{file_path}: {len(lines)} raw lines -> {len(kept)} kept sentences")
        cleaned.extend(kept)

    print(f"Dropped (not Amharic / bad / too short or long): {stats['bad']}")
    print(f"Dropped (duplicates and near-duplicates): {stats['dup']}")
    print(f"Total cleaned sentences: {len(cleaned)}")

    train, val = split_train_val(cleaned)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cleaned.txt").write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    (out_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

    print(f"Train: {len(train)} lines, Validation: {len(val)} lines")


if __name__ == "__main__":
    main()