"""SentencePiece wrapper utilities."""


def train_sentencepiece(
    input_path: str, model_type: str, vocab_size: int, output_prefix: str
) -> None:
    """Train a SentencePiece model."""
    raise NotImplementedError


def load_sentencepiece(model_path: str):
    """Load a trained SentencePiece model."""
    raise NotImplementedError
