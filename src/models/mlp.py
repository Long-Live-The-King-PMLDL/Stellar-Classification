import torch
import torch.nn as nn


class StellarMLP(nn.Module):
    """
    MLP for stellar object classification.

    Architecture:
        input
        -> [Linear -> BatchNorm -> ReLU -> Dropout] x N
        -> Linear(num_classes)

    The model returns raw logits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

            current_dim = hidden_dim

        layers.append(
            nn.Linear(current_dim, num_classes)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)