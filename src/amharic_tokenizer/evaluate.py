"""Evaluation utilities for comparing tokenizers."""


def tokens_per_word(tokenizer, text: str) -> float:
    """Compute average tokens per word."""
    raise NotImplementedError


def compression_ratio(tokenizer, text: str) -> float:
    """Compute compression ratio (bytes / tokens)."""
    raise NotImplementedError


def vocab_coverage(tokenizer, text: str) -> float:
    """Compute fraction of words covered by vocab."""
    raise NotImplementedError


def compare_tokenizers(tokenizers: dict, text: str) -> dict:
    """Compare multiple tokenizers on the same text."""
    raise NotImplementedError
