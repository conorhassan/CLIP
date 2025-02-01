
import torch 
from torch import nn, functional as F


class ConstrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(
        self, 
        text_features, 
        image_features
    ):
        # normalize the features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # compute logits
        logits = torch.matmul(text_features, image_features.T) / self.temperature

        targets = torch.arange(text_features.size(0))
        loss = nn.CrossEntropyLoss()(logits, targets) + nn.CrossEntropyLoss()(logits.T, targets) / 2
        return loss
