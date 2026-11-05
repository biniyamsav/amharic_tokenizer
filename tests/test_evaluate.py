from src.amharic_tokenizer.evaluate import tokens_per_word


def test_tokens_per_word_one_token_each():
    lines = ["ሰላም ነው", "እንዴት ነህ"]
    assert tokens_per_word(lambda line: line.split(), lines) == 1.0


def test_tokens_per_word_counts_extra_tokens():
    # every word becomes 2 tokens
    lines = ["ሰላም ነው"]
    assert tokens_per_word(lambda line: list(line.replace(" ", "")), lines) > 1.0


def test_tokens_per_word_exact_value():
    # 2 words, 3 tokens -> 1.5
    assert tokens_per_word(lambda line: ["a", "b", "c"], ["ሰላም ነው"]) == 1.5