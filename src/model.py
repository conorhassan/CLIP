import math 
import torch
from torch import nn, Tensor 
from einops import rearrange, repeat
from typing import Union, Optional, Tuple
from jaxtyping import Int, Float
from dataclasses import dataclass, field

@dataclass 
class TextConfig:
    max_position_dim: int = 77
    vocab_dim: int = 49408
    num_transformer_layers: int = 12
    embedding_dim: int = 512
    num_heads: int = 8

@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 32
    in_channels: int = 3
    num_transformer_layers: int = 12
    embedding_dim: int = 768
    num_heads: int = 12

    def __post_init__(self):
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

@dataclass
class CLIPConfig:
    text: TextConfig = field(default_factory=TextConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    projection_dim: int = 512

class CLIPModel(nn.Module):
    def __init__(
        self, 
        config: CLIPConfig
    ):
        super().__init__()
        self.text_model = TextModel(config.text)
        self.vision_model = VisionModel(config.vision)
        self.text_projection = nn.Linear(
            config.text.embedding_dim, 
            config.projection_dim, 
            bias=False
        )
        self.vision_projection = nn.Linear(
            config.vision.embedding_dim, 
            config.projection_dim, 
            bias=False
        )
    
    def forward(
        self, 
        text: Int[Tensor, "batch seq"], 
        image: Float[Tensor, "batch channels height width"], 
        mask: Float[Tensor, "batch seq"] = None
    ) -> Tuple[Float[Tensor, "batch projection_dim"]]:
        text = self.text_model(text, mask)
        text = self.text_projection(text.mean(dim=1)) # mean pool then project
        image = self.vision_model(image)
        image = self.vision_projection(image) # already pooled (cls token)
        return text, image

class TextModel(nn.Module):

    def __init__(
        self, 
        config: TextConfig
    ):
        super().__init__()
        self.embeddings = TextEmbedding(config)
        self.encoder = Encoder(config)
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
        config: TextConfig
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
    
class VisionModel(nn.Module):
    def __init__(
        self,
        config: VisionConfig
    ):
        super().__init__()
        self.embeddings = VisionEmbedding(config)
        self.encoder = Encoder(config)
        self.pre_layrnorm = nn.LayerNorm(config.embedding_dim) # typo in HF naming
        self.post_layernorm = nn.LayerNorm(config.embedding_dim)

    def forward(
        self, 
        x: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch embedding_dim"]:
        x = self.embeddings(x)
        x = self.pre_layrnorm(x)
        x = self.encoder(x, mask=None)
        x = self.post_layernorm(x[:, 0, :]) # select only cls token
        return x

class VisionEmbedding(nn.Module):

    def __init__(
        self, 
        config: VisionConfig
    ): 
        super().__init__()
        self.embedding_dim = config.embedding_dim 
        self.num_patches = config.num_patches
        self.num_positions = config.num_positions 
        self.patch_embedding = nn.Conv2d(
            in_channels=config.in_channels, 
            out_channels=config.embedding_dim, 
            kernel_size=config.patch_size, 
            stride=config.patch_size, 
            bias=False
        )
        self.class_embedding = nn.Parameter(torch.randn(config.embedding_dim))
        self.position_embedding = nn.Embedding(config.num_positions, config.embedding_dim)

    def forward(
        self, 
        x: Float[Tensor, "batch channels height width"], 
    ) -> Float[Tensor, "batch seq embedding_dim"]:
        x = self.patch_embedding(x) # b embedding_dim height/patch_size width/patch_size
        x = rearrange(x, "b c h w -> b (h w) c") # b num_patches embedding_dim
        cls_token = repeat(self.class_embedding, "c -> b 1 c", b=x.shape[0])
        x = torch.cat([cls_token, x], dim=1)
        positions = torch.arange(self.num_positions, device=x.device)
        x = x + self.position_embedding(positions)

        return x  

class Encoder(nn.Module):

    def __init__(
        self, 
        config: Union[TextConfig, VisionConfig]
    ): 
        super().__init__()
        self.layers = nn.ModuleList([
            Transformer(config) for _ in range(config.num_transformer_layers)
        ])

    def forward(
        self, 
        x: Float[Tensor, "batch seq embedding_dim"], 
        mask: Optional[Float[Tensor, "batch seq"]] = None, # NOTE: mask not required for vision transformer
    ) -> Float[Tensor, "batch seq embedding_dim"]:
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class Transformer(nn.Module):

    def __init__(
        self, 
        config: Union[TextConfig, VisionConfig]
    ):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim)

    def forward(
        self, 
        x: Float[Tensor, "batch seq embedding_dim"],
        mask: Optional[Float[Tensor, "batch seq"]] = None
    ) -> Float[Tensor, "batch seq embedding_dim"]:
        x = x + self.self_attn(self.layer_norm1(x), mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x 

class MLP(nn.Module):

    def __init__(
        self, 
        config: Union[TextConfig, VisionConfig]
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
        config: Union[TextConfig, VisionConfig]
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
        mask: Optional[Float[Tensor, "batch seq"]] = None
    ) -> Float[Tensor, "batch seq embedding_dim"]:
        key = self.k_proj(x)
        value = self.v_proj(x)
        query = self.q_proj(x)
        key = rearrange(key, "b s (h d) -> b h d s", h=self.num_heads)
        value = rearrange(value, "b s (h d) -> b h s d", h=self.num_heads)
        query = rearrange(query, "b s (h d) -> b h s d", h=self.num_heads)
        attention = torch.matmul(query, key) / math.sqrt(self.head_dim) # (b h s d) x (b h d s) -> (b h s s)
        if mask is not None:
            mask = repeat(mask, "b s -> b 1 s t", t=mask.shape[1]) # b s -> b 1 s s
            mask = (1.0 - mask) * torch.finfo(torch.float32).min # 1 -> 0, 0 -> -inf
            attention = attention + mask # attention after softmax = 0 when OG mask = 0
        attention = torch.softmax(attention, dim=-1)
        attention = torch.matmul(attention, value) # (b h s s) x (b h s d) -> b h s d
        attention = rearrange(attention, "b h s d -> b s (h d)") # b h s d -> b s e
        return self.out_proj(attention)