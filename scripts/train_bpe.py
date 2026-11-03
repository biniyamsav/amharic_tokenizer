"""Train a BPE tokenizer on the processed Amharic corpus."""
import argparse
import json
import time
from pathlib import Path

from src.amharic_tokenizer.bpe import train_bpe
from src.amharic_tokenizer.utils import read_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/train.txt")
    parser.add_argument("--output", default="models/bpe_merges.json")
    parser.add_argument("--num-merges", type=int, default=500)
    args = parser.parse_args()

    lines = read_corpus(args.input)
    print(f"Training on {len(lines)} lines, {args.num_merges} merges...")

    start = time.time()
    merges = train_bpe(lines, args.num_merges)
    print(f"Done in {time.time() - start:.1f}s")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merges, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(merges)} merges to {out}")
    print("First 10 merges:", merges[:10])


if __name__ == "__main__":
    main()