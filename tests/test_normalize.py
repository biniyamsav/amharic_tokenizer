from src.amharic_tokenizer.normalize import (
    clean_punctuation,
    normalize_text,
    space_punctuation,
    unify_homophones,
)


def test_homophones_first_form():
    assert unify_homophones("ሠላም") == "ሰላም"


def test_homophones_other_forms():
    assert unify_homophones("ሡ") == "ሱ"


def test_clean_punctuation():
    assert clean_punctuation("ሰላም፡ዓለም፡፡") == "ሰላም ዓለም።"


def test_normalize_text():
    assert normalize_text("  ሠላም፡፡   ሡ  ") == "ሰላም ። ሱ"


def test_homophones_can_be_turned_off():
    assert normalize_text("ሠላም", homophones=False) == "ሠላም"


def test_space_punctuation():
    assert space_punctuation("ሰላም ነው።").split() == ["ሰላም", "ነው", "።"]