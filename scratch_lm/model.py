"""A decoder-only Transformer implemented for the scratch LM experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 256
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    dropout: float = 0.1

    def validate(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        config = cls(**values)
        config.validate()
        return config


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_head,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.norm_mlp = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            values,
            values,
            values,
            attn_mask=mask,
            need_weights=False,
        )
        values = self.norm_attention(values + attended)
        return self.norm_mlp(values + self.mlp(values))


class ScratchTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.context_length, config.d_model
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, length = token_ids.shape
        if length > self.config.context_length:
            raise ValueError("sequence exceeds context_length")
        positions = torch.arange(length, device=token_ids.device)
        values = self.token_embedding(token_ids) + self.position_embedding(positions)
        values = self.dropout(values)
        mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=token_ids.device),
            diagonal=1,
        )
        for block in self.blocks:
            values = block(values, mask)
        return self.output(self.final_norm(values))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_id: int | None = None,
    ) -> list[int]:
        self.eval()
        generated = list(prompt_ids)
        device = next(self.parameters()).device
        for _ in range(max_new_tokens):
            context = generated[-self.config.context_length :]
            inputs = torch.tensor([context], dtype=torch.long, device=device)
            logits = self(inputs)[0, -1].float()
            if temperature == 0:
                next_id = int(torch.argmax(logits).item())
            else:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                if top_k:
                    values, indices = torch.topk(
                        probabilities, min(top_k, probabilities.numel())
                    )
                    choice = int(torch.multinomial(values, 1).item())
                    next_id = int(indices[choice].item())
                else:
                    next_id = int(torch.multinomial(probabilities, 1).item())
            generated.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
        return generated
