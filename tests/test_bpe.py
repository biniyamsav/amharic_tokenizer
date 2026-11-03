from src.amharic_tokenizer.bpe import (
    count_words,
    encode,
    encode_word,
    get_pair_counts,
    merge_pair,
    train_bpe,
)


def test_count_words():
    counts = count_words(["ሰላም ዓለም", "ሰላም ነው"])
    assert counts["ሰላም"] == 2
    assert counts["ዓለም"] == 1


def test_get_pair_counts():
    pairs = get_pair_counts({("ሰ", "ላ", "ም"): 2, ("ነ", "ው"): 1})
    assert pairs[("ሰ", "ላ")] == 2
    assert pairs[("ነ", "ው")] == 1


def test_merge_pair():
    result = merge_pair(("ሰ", "ላ"), {("ሰ", "ላ", "ም"): 2, ("ነ", "ው"): 1})
    assert result == {("ሰላ", "ም"): 2, ("ነ", "ው"): 1}


def test_train_bpe_order():
    merges = train_bpe(["ሰላም ሰላም ነው"], 2)
    assert merges == [("ሰ", "ላ"), ("ሰላ", "ም")]


def test_encode_word():
    merges = [("ሰ", "ላ"), ("ሰላ", "ም")]
    assert encode_word("ሰላም", merges) == ["ሰላም"]
    assert encode_word("ሰላም", merges[:1]) == ["ሰላ", "ም"]


def test_encode_sentence():
    merges = [("ሰ", "ላ"), ("ሰላ", "ም")]
    assert encode("ሰላም ነው", merges) == ["ሰላም", "ነ", "ው"]