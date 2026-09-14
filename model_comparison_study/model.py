"""
author:Shuaifeng
data:10/8/2022
"""
import numpy as np
import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.name = 'FullyConnected'

    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)
        y = self.relu(y)
        y = self.classifier(y)
        return y


class TinyMLP(nn.Module):
    """Cheap end of the comparison: ~10-20x fewer params than FullyConnected."""
    def __init__(self, num_classes, hidden_size=32):
        super(TinyMLP, self).__init__()
        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.name = 'TinyMLP'

    def forward(self, x):
        y = self.relu(self.fc1(x))
        y = self.relu(self.fc2(y))
        return self.classifier(y)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.fc(x)
        y = self.norm(y)
        y = self.relu(y)
        y = self.dropout(y)
        return x + y


class ResMLP(nn.Module):
    """Same width as FullyConnected, but with residual connections, LayerNorm,
    and dropout actually enabled (FullyConnected defines a dropout layer it
    never calls in forward())."""
    def __init__(self, num_classes, hidden_size=256, num_blocks=3, dropout=0.2):
        super(ResMLP, self).__init__()
        self.input_proj = nn.Linear(4, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)])
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.name = 'ResMLP'

    def forward(self, x):
        y = self.relu(self.input_proj(x))
        for block in self.blocks:
            y = block(y)
        return self.classifier(y)


class FTTransformer(nn.Module):
    """Tabular transformer: each of the 4 input features becomes its own
    token (plus a learned CLS token), self-attention mixes across them, and
    the CLS token is classified. The expensive end of the comparison -- with
    only 4 feature-tokens, attention is unlikely to buy much over ResMLP,
    which is itself part of the point for a resource-efficiency study."""
    def __init__(self, num_classes, d_model=32, num_heads=4, num_layers=2,
                 dim_feedforward=64, dropout=0.1):
        super(FTTransformer, self).__init__()
        self.num_features = 4
        self.feature_embed = nn.Linear(1, d_model)
        self.feature_type_embed = nn.Parameter(torch.randn(1, self.num_features, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.name = 'FTTransformer'

    def forward(self, x):
        b = x.shape[0]
        tokens = self.feature_embed(x.unsqueeze(-1))  # (batch, 4, d_model)
        tokens = tokens + self.feature_type_embed
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, 5, d_model)
        encoded = self.encoder(tokens)
        return self.classifier(encoded[:, 0])


MODEL_REGISTRY = {
    'mlp': FullyConnected,
    'tinymlp': TinyMLP,
    'resmlp': ResMLP,
    'fttransformer': FTTransformer,
}
