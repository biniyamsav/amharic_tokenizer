from collections import Counter
import re

def count_words(lines: list[str]) -> Counter:
    """Count how many times each word appears in the text."""
    counts = Counter()
    for line in lines:
        counts.update(pre_tokenize(line))
    return counts
def get_pair_counts(word_freqs: dict) -> Counter:
    """Count how often each neighboring pair of tokens appears."""
    pairs = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs
def merge_pair(pair: tuple, word_freqs: dict) -> dict:
    """Merge every occurrence of `pair` into a single token in all words."""
    merged_token = pair[0] + pair[1]
    new_freqs = {}
    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_freqs[tuple(new_word)] = freq
    return new_freqs
def train_bpe(lines: list[str], num_merges: int) -> list[tuple]:
    """Learn BPE merges from text. Returns the merges in the order they were learned."""
    word_counts = count_words(lines)
    word_freqs = {tuple(word): freq for word, freq in word_counts.items()}

    merges = []
    for _ in range(num_merges):
        pairs = get_pair_counts(word_freqs)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        word_freqs = merge_pair(best, word_freqs)
        merges.append(best)
    return merges
def encode_word(word: str, merges: list[tuple]) -> list[str]:
    """Split one word into BPE tokens by applying the merges in order."""
    tokens = list(word)
    for pair in merges:
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i + 1]) == pair:
                tokens[i:i + 2] = [pair[0] + pair[1]]
            else:
                i += 1
    return tokens
def encode(text: str, merges: list[tuple]) -> list[str]:
    """Tokenize a full line of text into BPE tokens."""
    tokens = []
    for word in pre_tokenize(text):
        tokens.extend(encode_word(word, merges))
    return tokens
def pre_tokenize(text: str) -> list[str]:
    """Split text into words and single punctuation marks."""
    return re.findall(r"\w+|[^\w\s]", text)