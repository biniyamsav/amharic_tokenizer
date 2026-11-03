"""From-scratch BPE tokenizer for Amharic."""


class BPETokenizer:
    """Byte Pair Encoding tokenizer implemented from scratch."""

    def train(self, corpus: str, vocab_size: int) -> None:
        """Train the BPE model on the given corpus."""
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back into text."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save the tokenizer to disk."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a tokenizer from disk."""
        raise NotImplementedError
