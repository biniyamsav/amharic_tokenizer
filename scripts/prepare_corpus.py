"""Clean the raw corpus and split it into train/validation files."""
from pathlib import Path

from src.amharic_tokenizer.normalize import normalize_text
from src.amharic_tokenizer.utils import (
    ethiopic_char_ratio,
    read_corpus,
    split_train_val,
)

MIN_RATIO = 0.8   # keep lines that are at least 80% Amharic letters
MIN_CHARS = 10    # drop very short lines (headings, fragments)


def main():
    lines = read_corpus("data/raw/amwiki.txt")
    print(f"Raw lines: {len(lines)}")

    cleaned = []
    for line in lines:
        if ethiopic_char_ratio(line) < MIN_RATIO:
            continue
        line = normalize_text(line)
        if len(line) >= MIN_CHARS:
            cleaned.append(line)
    print(f"Kept lines: {len(cleaned)}")

    train, val = split_train_val(cleaned)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    print(f"Train: {len(train)} lines, Val: {len(val)} lines")


if __name__ == "__main__":
    main()