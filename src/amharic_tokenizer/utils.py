"""Utility functions."""


def read_corpus(path: str) -> str:
    """Read a corpus file and return its text."""
    raise NotImplementedError


def split_train_val(text: str, val_ratio: float = 0.1) -> tuple[str, str]:
    """Split text into train and validation parts."""
    raise NotImplementedError


def ethiopic_char_ratio(text: str) -> float:
    """Compute ratio of Ethiopic characters in text."""
    raise NotImplementedError
