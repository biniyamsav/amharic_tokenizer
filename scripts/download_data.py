
"""Download a limited Amharic pretraining corpus."""

import time
from pathlib import Path
from datasets import load_dataset

TARGET_CHARS = 50_000_000


def main():
    out = Path("data/raw/amharic_pretraining.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "yordanoswuletaw/amharic-pretraining-corpus",
        split="train",
        streaming=True,
    )

    ds = ds.shuffle(seed=42, buffer_size=10_000)

    total_chars = 0
    total_samples = 0
    skipped = 0

    with open(out, "w", encoding="utf-8") as f:
        for sample in ds:
            text = sample.get("text")

            # Skip missing or invalid text
            if not isinstance(text, str) or not text.strip():
                skipped += 1
                continue

            text = text.strip()

            f.write(text + "\n")

            total_chars += len(text)
            total_samples += 1

            if total_samples % 1000 == 0:
                print(
                    f"Characters: {total_chars:,} / {TARGET_CHARS:,} | "
                    f"Samples: {total_samples:,}"
                )

            if total_chars >= TARGET_CHARS:
                break

    print("\nDownload complete!")
    print(f"Characters: {total_chars:,}")
    print(f"Samples: {total_samples:,}")
    print(f"Skipped: {skipped:,}")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
