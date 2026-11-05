"""Evaluate a trained tokenizer on the validation set."""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.amharic_tokenizer.evaluate import roundtrip_failures, tokens_per_word
from src.amharic_tokenizer.sentencepiece_wrapper import load_tokenizer, tokenize
from src.amharic_tokenizer.utils import read_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/am_bpe_20k.model")
    parser.add_argument("--val", default="data/processed/val.txt")
    args = parser.parse_args()

    sp = load_tokenizer(args.model)
    val = read_corpus(args.val)

    print(f"Model: {args.model}")
    print(f"Vocabulary size: {sp.get_piece_size()}")
    print(f"Validation lines: {len(val)}")

    tpw = tokens_per_word(lambda line: tokenize(sp, line), val)
    print(f"Tokens per word: {tpw:.3f}")

    failures = roundtrip_failures(sp, val)
    print(f"Round-trip failures: {len(failures)} of {len(val)}")
    for original, result in failures[:5]:
        print("  original:", original)
        print("  result:  ", result)


if __name__ == "__main__":
    main()