"""Train a SentencePiece tokenizer on the processed Amharic corpus."""
import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.amharic_tokenizer.sentencepiece_wrapper import train_sentencepiece


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/train.txt")
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    prefix = f"{args.output_dir}/am_{args.model_type}_{args.vocab_size // 1000}k"

    print(f"Training {args.model_type} with vocab size {args.vocab_size}...")
    start = time.time()
    train_sentencepiece(args.input, prefix, args.vocab_size, args.model_type)
    print(f"Done in {time.time() - start:.1f}s. Saved {prefix}.model")


if __name__ == "__main__":
    main()