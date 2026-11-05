from pathlib import Path

import pytest

from src.amharic_tokenizer.normalize import normalize_text
from src.amharic_tokenizer.sentencepiece_wrapper import load_tokenizer, tokenize

MODEL = "models/am_bpe_20k.model"

pytestmark = pytest.mark.skipif(
    not Path(MODEL).exists(), reason="trained model not found"
)


@pytest.fixture(scope="module")
def sp():
    return load_tokenizer(MODEL)


def test_vocab_size(sp):
    assert sp.get_piece_size() == 20000


def test_common_word_is_one_token(sp):
    assert tokenize(sp, "ሰላም") == ["▁ሰላም"]


def test_punctuation_is_separate(sp):
    tokens = tokenize(sp, "ሰላም ነው።")
    assert "▁።" in tokens


def test_homophone_spelling_gives_same_tokens(sp):
    assert tokenize(sp, "ሠላም") == tokenize(sp, "ሰላም")


def test_empty_string(sp):
    assert tokenize(sp, "") == []


@pytest.mark.parametrize("text", [
    "hello world",
    "ሰላም 😀 ነው",
    "ሰላም ☃ ነው",
    "12345",
    "ሰላም\nነው",
])
def test_odd_input_never_crashes_or_loses_text(sp, text):
    ids = sp.encode(normalize_text(text))
    assert sp.decode(ids) == normalize_text(text)