import re
import unicodedata
def unify_homophones(text: str) -> str:
    """Replace duplicate-sound letters (all 7 forms) with one standard letter."""
    # (duplicate letter, standard letter)
    pairs = ["ሐሀ", "ኀሀ", "ሠሰ", "ዐአ", "ፀጸ"]
    for pair in pairs:
        old_start = ord(pair[0])
        new_start = ord(pair[1])
        for i in range(7):
            text = text.replace(chr(old_start + i), chr(new_start + i))
    return text
def clean_punctuation(text: str) -> str:
    """Turn the old word separator into a space and fix double separators."""
    text = text.replace("፡፡", "።")
    text = text.replace("፡", " ")
    return text
def normalize_text(text: str, homophones: bool = True) -> str:
    """Clean Amharic text: standard Unicode form, homophones, punctuation, spaces."""
    text = unicodedata.normalize("NFC", text)
    if homophones:
        text = unify_homophones(text)
    text = clean_punctuation(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text