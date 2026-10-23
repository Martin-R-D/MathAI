"""Versioned, JSON-serializable architecture configuration."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 264
    embedding_dim: int = 128
    attention_heads: int = 4
    encoder_layers: int = 4
    feedforward_dim: int = 512
    dropout: float = 0.1
    max_sequence_length: int = 2048
    format_version: int = 1

    def __post_init__(self):
        for name in ("vocab_size", "embedding_dim", "attention_heads", "encoder_layers",
                     "feedforward_dim", "max_sequence_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.embedding_dim % self.attention_heads:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        if (type(self.dropout) not in (int, float) or not math.isfinite(self.dropout)
                or not 0 <= self.dropout < 1):
            raise ValueError("dropout must be finite and in [0, 1)")
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError("Unsupported model configuration version")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
