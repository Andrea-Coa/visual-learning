"""MLP and CNN architectures from Rauber et al. (2017), Sec. 4.

Both models expose their hidden-layer activations: `model(x, return_hidden=True)` returns
`(logits, {layer_name: activations [N, features]})`. Activations are taken after the ReLU
(and pooling, for conv blocks) and before dropout, i.e. what the next layer receives at test time.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .config import DatasetSpec


def glorot_init(module: nn.Module) -> None:
    """Glorot-uniform weights and zero biases, as in the paper."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


class MLP(nn.Module):
    """Four ReLU hidden layers of 1000 units, dropout growing 0.2 -> 0.5, softmax output."""

    def __init__(self, in_features: int, n_classes: int, width: int = 1000,
                 dropouts: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5)):
        super().__init__()
        dims = [in_features] + [width] * len(dropouts)
        self.layer_names = tuple(f"h{i + 1}" for i in range(len(dropouts)))
        self.hidden = nn.ModuleList(nn.Linear(d_in, d_out) for d_in, d_out in zip(dims, dims[1:]))
        self.dropouts = nn.ModuleList(nn.Dropout(p) for p in dropouts)
        self.out = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        h = x.flatten(1)
        hidden = {}
        for name, linear, dropout in zip(self.layer_names, self.hidden, self.dropouts):
            h = F.relu(linear(h))
            hidden[name] = h
            h = dropout(h)
        logits = self.out(h)
        return (logits, hidden) if return_hidden else logits


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 'same' convolutions with ReLU, followed by 2x2 max pooling."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
    )


class CNN(nn.Module):
    """conv32-conv32-pool-conv64-conv64-pool-fc512-softmax (Keras 'cifar10_cnn' style).

    Hidden layers recorded:
      conv1: output of the first block, flattened (32*14*14 = 6272 for MNIST, 8192 otherwise)
      conv2: output of the second block, flattened (3136 / 4096; the paper's "4096 (or 3136)" layer)
      fc:    the 512-unit fully connected layer
    """

    layer_names = ("conv1", "conv2", "fc")

    def __init__(self, channels: int, size: int, n_classes: int):
        super().__init__()
        self.block1 = conv_block(channels, 32)
        self.block2 = conv_block(32, 64)
        self.fc = nn.Linear(64 * (size // 4) ** 2, 512)
        self.out = nn.Linear(512, n_classes)
        self.drop_conv = nn.Dropout(0.25)
        self.drop_fc = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        b1 = self.block1(x)
        b2 = self.block2(self.drop_conv(b1))
        fc = F.relu(self.fc(self.drop_conv(b2).flatten(1)))
        logits = self.out(self.drop_fc(fc))
        if not return_hidden:
            return logits
        return logits, {"conv1": b1.flatten(1), "conv2": b2.flatten(1), "fc": fc}


def build_model(name: str, spec: DatasetSpec) -> nn.Module:
    if name == "mlp":
        model = MLP(spec.channels * spec.size ** 2, spec.n_classes)
    elif name == "cnn":
        model = CNN(spec.channels, spec.size, spec.n_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    glorot_init(model)
    return model
