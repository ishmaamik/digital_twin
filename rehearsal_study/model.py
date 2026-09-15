"""
author:Shuaifeng
data:10/8/2022
"""
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


MODEL_REGISTRY = {
    'mlp': FullyConnected,
    'tinymlp': TinyMLP,
}
