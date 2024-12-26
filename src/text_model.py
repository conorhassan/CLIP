import math 
import torch
from torch import nn, Tensor
from dataclasses import dataclass
from jaxtyping import Int, Float
from einops import rearrange, repeat


@dataclass 
class TextModelConfig:
    max_position_dim: int = 77
    vocab_dim: int = 49408
    num_transformer_layers: int = 12
    embedding_dim: int = 512
    num_heads: int = 8


class TextModel(nn.Module):

    def __init__(
        self, 
        config: TextModelConfig
    ):
        super().__init__()
        self.embeddings = TextEmbedding(config)
        self.encoder = TextEncoder(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dim)

    def forward(
        self, 
        x: Float[Tensor, "batch seq"], 
        mask: Float[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq embedding_dim"]:
        x = self.embeddings(x)
        x = self.encoder(x, mask)
        return self.final_layer_norm(x) 
    

class TextEmbedding(nn.Module):

    def __init__(
        self, 
        config: TextModelConfig
    ):
        super().__init__()
        self.max_position_dim = config.max_position_dim
        self.embedding_dim = config.embedding_dim
        self.vocab_dim = config.vocab_dim 
        self.token_embedding = nn.Embedding(self.vocab_dim, self.embedding_dim)
        self.position_embedding = nn.Embedding(self.max_position_dim, self.embedding_dim)

    def forward(
        self, 
        x: Int[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq embedding_dim"]:
        token_embed = self.token_embedding(x)
        position_ids = torch.arange(x.shape[1], device=x.device)
        position_embed = self.position_embedding(position_ids)
        return token_embed + position_embed


class TextEncoder(nn.Module):

    def __init__(
        self, 
        config: TextModelConfig
    ): 
        super().__init__()
        self.layers = nn.ModuleList([
            Transformer(config) for _ in range(config.num_transformer_layers)
        ])

    def forward(
        self, 
        x: Float[Tensor, "batch seq embedding_dim"], 
        mask: Float[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq embedding_dim"]:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Transformer(nn.Module):

    def __init__(
        self, 
        config: TextModelConfig
    ):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim)

    def forward(
        self, 
        x: Float[Tensor, "batch seq embedding_dim"],
        mask: Float[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq embedding_dim"]:
        x = x + self.self_attn(self.layer_norm1(x), mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x 
    

class MLP(nn.Module):

    def __init__(
        self, 
        config: TextModelConfig
    ): 
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.fc1 = nn.Linear(self.embedding_dim, 4 * self.embedding_dim)
        self.fc2 = nn.Linear(4 * self.embedding_dim, self.embedding_dim)
        self.gelu = nn.GELU()

    def forward(
        self, 
        x: Float[Tensor, "batch seq embedding_dim"]
    ) -> Float[Tensor, "batch seq embedding_dim"]:
        x = self.gelu(self.fc1(x))
        return self.fc2(x)


class Attention(nn.Module):

    def __init__(
        self, 
        config: TextModelConfig
    ):
        super().__init__()
        self.embedding_dim = config.embedding_dim 
        self.num_heads = config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads
        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(
        self, 
        x: Float[Tensor, "batch seq embedding_dim"], 
        mask: Float[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq embedding_dim"]:
        key = self.k_proj(x)
        value = self.v_proj(x)
        query = self.q_proj(x)
        key = rearrange(key, "b s (h d) -> b h d s", h=self.num_heads)
        value = rearrange(value, "b s (h d) -> b h s d", h=self.num_heads)
        query = rearrange(query, "b s (h d) -> b h s d", h=self.num_heads)
        mask = repeat(mask, "b s -> b 1 s t", t=mask.shape[1]) # b s -> b 1 s s
        mask = (1.0 - mask) * torch.finfo(torch.float32).min # 1 -> 0, 0 -> -inf
        attention = torch.matmul(query, key) / math.sqrt(self.head_dim)
        attention = attention + mask # attention after softmax = 0 when OG mask = 0
        attention = torch.softmax(attention, dim=-1)
        attention = torch.matmul(attention, value)
        attention = rearrange(attention, "b h s d -> b s (h d)")
        return self.out_proj(attention)