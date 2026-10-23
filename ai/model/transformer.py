"""A small encoder scoring each serialized candidate independently."""

import torch
from torch import nn

from .config import ModelConfig


class ActionTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.positions = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        layer = nn.TransformerEncoderLayer(
            config.embedding_dim, config.attention_heads, config.feedforward_dim,
            config.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.encoder_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, 1)
        # Independently initialize cloned encoder layers.
        for name, parameter in self.named_parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
            elif name.endswith("weight"):
                nn.init.ones_(parameter)
            else:
                nn.init.zeros_(parameter)
        with torch.no_grad():
            self.tokens.weight[0].zero_()

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, input_ids, attention_mask):
        if (input_ids.ndim != 2 or input_ids.shape != attention_mask.shape
                or input_ids.shape[0] == 0 or input_ids.shape[1] == 0):
            raise ValueError("Expected nonempty [batch, sequence] IDs and matching mask")
        if input_ids.dtype != torch.long or attention_mask.dtype != torch.bool:
            raise ValueError("IDs must be int64 and attention_mask boolean")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("Sequence exceeds model maximum length")
        if bool(((input_ids < 0) | (input_ids >= self.config.vocab_size)).any()):
            raise ValueError("Token ID outside vocabulary")
        if not torch.equal(attention_mask, input_ids != 0):
            raise ValueError("Mask must be true exactly for nonpadding tokens")
        if not bool(attention_mask.any(dim=1).all()):
            raise ValueError("Every sequence must contain a nonpadding token")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.tokens(input_ids) + self.positions(positions)[None, :, :]
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden*mask).sum(dim=1) / mask.sum(dim=1)
        return self.head(self.norm(pooled)).squeeze(-1)
