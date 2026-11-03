"""Text normalization utilities for Amharic."""


def normalize_text(text: str) -> str:
    """Normalize Amharic text (Unicode normalization, etc.)."""
    raise NotImplementedError


def unify_homophones(text: str) -> str:
    """Unify visually/phonetically similar Amharic characters."""
    raise NotImplementedError


def clean_punctuation(text: str) -> str:
    """Clean and normalize punctuation in text."""
    raise NotImplementedError


def strip_non_ethiopic(text: str) -> str:
    """Remove characters that are not Ethiopic script."""
    raise NotImplementedError
