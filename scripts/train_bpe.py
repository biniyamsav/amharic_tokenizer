"""Train BPE tokenizer CLI."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
