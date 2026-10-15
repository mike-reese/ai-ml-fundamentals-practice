import torch
from torch import nn


def linear(layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """x @ W^T + b for a dense or a sparse x. Sparse rows skip the zeros, which are 99.99% of TF-IDF."""
    if x.is_sparse or x.layout == torch.sparse_csr:
        return torch.sparse.mm(x, layer.weight.t()) + layer.bias
    return layer(x)

class MultinomialLogisticRegression(nn.Module):
    def __init__(self, n_features: int, n_classes: int=3):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(self.linear, x)


class OneHiddenLayerMLP(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, n_classes: int=3):
        super().__init__()
        self.hidden = nn.Linear(n_features, hidden_size)
        self.out = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(linear(self.hidden, x)))

