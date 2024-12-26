import torch
from torch import nn, Tensor
from dataclasses import dataclass
from jaxtyping import Int, Float

@dataclass
class TextEmbeddingConfig:
    max_position_dim: int = 77
    embedding_dim: int = 512 
    vocab_dim: int = 49408

class TextEmbedding(nn.Module):

    def __init__(self, config: TextEmbeddingConfig):
        super().__init__()
        self.max_position = config.max_position
        self.embedding_dim = config.embedding_dim
        self.vocab_dim = config.vocab_dim 
        self.token_embeddings = nn.Embedding(self.vocab_dim, self.embedding_dim)
        self.position_embeddings = nn.Embedding(self.max_position, self.embedding_dim)

    def forward(
        self, x: Int[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq embedding_dim"]:
        token_embed = self.token_embeddings(x)
        position_ids = torch.arange(x.shape[1], device=x.device)
        position_embed = self.position_embeddings(position_ids)
        return token_embed + position_embed

class TextModel(nn.Module):

    def __init__(self, embedding_config: TextEmbeddingConfig):
        super().__init__()
        self.embeddings = TextEmbedding(embedding_config)

    def forward(
        self, x: Int[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq embedding_dim"]:
        return self.embeddings(x)
    
class CLIP(nn.Module):

    def __init__(self, text_embedding_config: TextEmbeddingConfig):
        super().__init__()
        self.text_model = TextModel(text_embedding_config)

    def forward(
        self, x: Int[Tensor]
    )