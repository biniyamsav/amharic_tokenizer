"""Download Amharic Wikipedia text into data/raw/."""
from pathlib import Path

from datasets import load_dataset


def main():
    out = Path("data/raw/amwiki.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("wikimedia/wikipedia", "20231101.am", split="train")

    with open(out, "w", encoding="utf-8") as f:
        for article in ds:
            for line in article["text"].split("\n"):
                line = line.strip()
                if line:
                    f.write(line + "\n")

    print(f"Saved to {out}")


if __name__ == "__main__":
    main()