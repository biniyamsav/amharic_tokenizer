"""Metrics for judging a tokenizer."""


def tokens_per_word(tokenize_fn, lines: list[str]) -> float:
    """Average number of tokens per word over a list of lines."""
    total_tokens = 0
    total_words = 0
    for line in lines:
        total_words += len(line.split())
        total_tokens += len(tokenize_fn(line))
    return total_tokens / total_words
def roundtrip_failures(sp, lines: list[str]) -> list[tuple[str, str]]:
    """Encode then decode each line. Return the (original, result) pairs that differ."""
    failures = []
    for line in lines:
        back = sp.decode(sp.encode(line))
        if back != line:
            failures.append((line, back))
    return failures