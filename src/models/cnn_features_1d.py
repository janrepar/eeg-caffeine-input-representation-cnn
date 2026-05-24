import torch
import torch.nn as nn


class CNNFeatures1D(nn.Module):
    """
    1D CNN for EEG feature representation.

    Expected input: (batch_size, n_channels, n_features)
    Example: (batch_size, 32, 13)
    Output: logits with shape (batch_size, n_classes)
    Use with: nn.CrossEntropyLoss()
    """

    def __init__(
        self,
        n_channels: int,
        n_features: int,
        n_classes: int = 2,
        conv1_filters: int = 32,
        conv2_filters: int = 64,
        kernel_size_1: int = 3,
        kernel_size_2: int = 3,
        hidden_units: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=conv1_filters,
                kernel_size=kernel_size_1,
                padding=kernel_size_1 // 2,
                bias=False,
            ),
            nn.BatchNorm1d(conv1_filters),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Conv1d(
                in_channels=conv1_filters,
                out_channels=conv2_filters,
                kernel_size=kernel_size_2,
                padding=kernel_size_2 // 2,
                bias=False,
            ),
            nn.BatchNorm1d(conv2_filters),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_features)
            flattened_size = self.feature_extractor(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, hidden_units),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        logits = self.classifier(x)
        return logits