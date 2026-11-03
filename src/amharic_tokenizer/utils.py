from pathlib import Path
import random


def read_corpus(path: str) -> list[str]:
    """Read a text file and return its non-empty lines as a list."""
    lines = []
    with open(Path(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines
def split_train_val(
    lines: list[str], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Shuffle lines and split them into (train, validation) lists."""
    lines = lines.copy()
    random.Random(seed).shuffle(lines)
    n_val = int(len(lines) * val_ratio)
    return lines[n_val:], lines[:n_val]
def ethiopic_char_ratio(text: str) -> float:
    """Return the fraction of letters that are Ethiopic (0.0 to 1.0).

    Digits, spaces and punctuation are ignored.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    ethiopic = [c for c in letters if "\u1200" <= c <= "\u137f"]
    return len(ethiopic) / len(letters)