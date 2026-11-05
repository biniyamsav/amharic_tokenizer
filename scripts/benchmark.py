"""Benchmark our tokenizers against well-known baselines on held-out Amharic text."""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import sentencepiece as spm
import tiktoken
from huggingface_hub import hf_hub_download

from src.amharic_tokenizer.sentencepiece_wrapper import load_tokenizer, tokenize
from src.amharic_tokenizer.utils import read_corpus


def build_tokenizers(model_paths):
    """Return {name: (vocab_size, encode(text)->ids, encode_word(word)->ids)}."""
    toks = {}

    for path in model_paths:
        sp = load_tokenizer(path)
        # our tokenizer: normalization is part of it, so use tokenize()
        enc = lambda t, sp=sp: sp.encode(t)  # val text is already normalized
        toks[f"ours ({Path(path).stem})"] = (sp.get_piece_size(), enc, enc)

    xlmr_path = hf_hub_download("xlm-roberta-base", "sentencepiece.bpe.model")
    xsp = spm.SentencePieceProcessor(model_file=xlmr_path)
    enc = lambda t: xsp.encode(t)
    toks["XLM-R (250k)"] = (xsp.get_piece_size(), enc, enc)

    for name in ["cl100k_base", "o200k_base"]:
        tk = tiktoken.get_encoding(name)
        toks[f"{name}"] = (
            tk.n_vocab,
            lambda t, tk=tk: tk.encode(t, disallowed_special=()),
            lambda w, tk=tk: tk.encode(" " + w, disallowed_special=()),
        )
    return toks


def evaluate(vocab_size, encode, encode_word, lines):
    words = Counter(w for line in lines for w in line.split())
    n_words = sum(words.values())
    n_chars = sum(len(line) for line in lines)
    n_bytes = sum(len(line.encode("utf-8")) for line in lines)

    n_tokens = 0
    used = set()
    for line in lines:
        ids = encode(line)
        n_tokens += len(ids)
        used.update(ids)

    single = four_plus = 0
    for word, count in words.items():
        k = len(encode_word(word))
        if k == 1:
            single += count
        if k >= 4:
            four_plus += count

    return {
        "vocab": vocab_size,
        "tokens/word": n_tokens / n_words,
        "chars/token": n_chars / n_tokens,
        "bytes/token": n_bytes / n_tokens,
        "1-token words %": 100 * single / n_words,
        "4+ token words %": 100 * four_plus / n_words,
        "vocab used %": 100 * len(used) / vocab_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["models/am_bpe_20k.model"])
    parser.add_argument("--val", default="data/processed/val.txt")
    parser.add_argument("--out", default="results/benchmark.md")
    args = parser.parse_args()

    lines = read_corpus(args.val)
    print(f"Validation lines: {len(lines)}")

    rows = {}
    for name, (vocab, enc, enc_word) in build_tokenizers(args.models).items():
        print(f"Evaluating {name}...")
        rows[name] = evaluate(vocab, enc, enc_word, lines)

    cols = list(next(iter(rows.values())).keys())
    header = "| Tokenizer | " + " | ".join(cols) + " |"
    divider = "|---" * (len(cols) + 1) + "|"
    table = [header, divider]
    for name, r in rows.items():
        cells = [f"{r[c]:,}" if c == "vocab" else f"{r[c]:.2f}" for c in cols]
        table.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(table)

    print("\n" + text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()