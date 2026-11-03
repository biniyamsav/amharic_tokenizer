from src.amharic_tokenizer.utils import (
    ethiopic_char_ratio,
    read_corpus,
    split_train_val,
)


def test_split_sizes():
    train, val = split_train_val(list(range(20)))
    assert len(train) == 18
    assert len(val) == 2


def test_split_has_no_overlap():
    train, val = split_train_val(list(range(20)))
    assert set(train).isdisjoint(val)


def test_ratio_pure_amharic():
    assert ethiopic_char_ratio("ሰላም") == 1.0


def test_ratio_ignores_digits():
    assert ethiopic_char_ratio("ሰላም 2024") == 1.0


def test_ratio_english_is_zero():
    assert ethiopic_char_ratio("hello") == 0.0


def test_ratio_empty_line():
    assert ethiopic_char_ratio("") == 0.0


def test_read_corpus_skips_empty_lines(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("ሰላም\n\nዓለም\n", encoding="utf-8")
    assert read_corpus(str(f)) == ["ሰላም", "ዓለም"]