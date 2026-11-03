"""Train SentencePiece tokenizer CLI."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["bpe", "unigram"], required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
