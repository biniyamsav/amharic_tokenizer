"""Load a trained SentencePiece model and tokenize Amharic text."""

import re

import sentencepiece as spm

from .normalize import normalize_text

# normalize_text puts spaces around Ethiopic punctuation ("ሰላም ።").
# On the way back out we glue it to the previous word again ("ሰላም።").
_SPACE_BEFORE_PUNCT = re.compile(r" ([።፣፤፥፦፧፨])")


def load_tokenizer(model_path: str) -> spm.SentencePieceProcessor:
    """Load a trained SentencePiece model from disk."""
    return spm.SentencePieceProcessor(model_file=str(model_path))


def tokenize(sp: spm.SentencePieceProcessor, text: str) -> list[str]:
    """Normalize the text first, then split it into tokens."""
    return sp.encode(normalize_text(text), out_type=str)


class AmharicTokenizer:
    """Normalization + SentencePiece in one object (text <-> pieces <-> ids)."""

    def __init__(self, model_path: str) -> None:
        self.sp = load_tokenizer(model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    def tokenize(self, text: str) -> list[str]:
        """Text -> token strings."""
        return tokenize(self.sp, text)

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """Text -> token ids."""
        return self.sp.encode(
            normalize_text(text), out_type=int, add_bos=add_bos, add_eos=add_eos
        )

    def decode(self, ids: list[int], tidy: bool = True) -> str:
        """Token ids -> text.

        The result is the NORMALIZED form of the input (homophones merged,
        whitespace cleaned), not the original string.
        """
        text = self.sp.decode(list(ids))
        if tidy:
            text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        return text
def train_sentencepiece(
    input_path: str,
    model_prefix: str,
    vocab_size: int,
    model_type: str = "bpe",
) -> None:
    """Train a SentencePiece model on a text file (one sentence per line)."""
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        byte_fallback=True,
        minloglevel=2,
    )