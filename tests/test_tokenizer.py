from pathlib import Path

import pytest

from src.amharic_tokenizer.sentencepiece_wrapper import AmharicTokenizer

MODEL = "models/am_bpe_32k.model"

pytestmark = pytest.mark.skipif(
    not Path(MODEL).exists(), reason="trained model not found"
)


@pytest.fixture(scope="module")
def tok():
    return AmharicTokenizer(MODEL)


def test_vocab_size(tok):
    assert tok.vocab_size == 32000


def test_common_word_is_one_token(tok):
    assert tok.tokenize("ሰላም") == ["▁ሰላም"]


def test_punctuation_is_separate(tok):
    assert "▁።" in tok.tokenize("ሰላም ነው።")


def test_homophone_spellings_match(tok):
    assert tok.encode("ሠላም") == tok.encode("ሰላም")


def test_encode_decode_roundtrip(tok):
    text = "እንዴት ነህ ?"
    assert tok.decode(tok.encode(text)) == text


def test_empty_string(tok):
    assert tok.encode("") == []


@pytest.mark.parametrize("text", [
    "hello world",
    "ሰላም 😀 ነው",
    "ሰላም ☃ ነው",
    "12345",
    "ሰላም\nነው",
])
def test_odd_input_never_crashes(tok, text):
    ids = tok.encode(text)
    assert isinstance(tok.decode(ids), str)


def test_emoji_is_not_unknown(tok):
    # byte fallback: an unseen character must not become <unk> (id 0)
    assert 0 not in tok.encode("ሰላም 😀")