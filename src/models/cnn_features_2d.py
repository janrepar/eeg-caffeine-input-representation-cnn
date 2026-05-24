import torch
import torch.nn as nn


class CNNFeatures2D(nn.Module):
    """
    2D CNN for EEG feature representation.

    Expected input: (batch_size, 1, n_channels, n_features)
    Example: (batch_size, 1, 32, 13)
    Output: logits with shape (batch_size, n_classes)
    Use with: nn.CrossEntropyLoss()
    """

    def __init__(
        self,
        n_channels: int,
        n_features: int,
        n_classes: int = 2,
        conv1_filters: int = 16,
        conv2_filters: int = 32,
        kernel_size_1: tuple[int, int] = (3, 3),
        kernel_size_2: tuple[int, int] = (3, 3),
        pool_size_1: tuple[int, int] = (2, 1),
        pool_size_2: tuple[int, int] = (2, 1),
        hidden_units: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=conv1_filters,
                kernel_size=kernel_size_1,
                padding=(kernel_size_1[0] // 2, kernel_size_1[1] // 2),
                bias=False,
            ),
            nn.BatchNorm2d(conv1_filters),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=pool_size_1),
            nn.Dropout(dropout),

            nn.Conv2d(
                in_channels=conv1_filters,
                out_channels=conv2_filters,
                kernel_size=kernel_size_2,
                padding=(kernel_size_2[0] // 2, kernel_size_2[1] // 2),
                bias=False,
            ),
            nn.BatchNorm2d(conv2_filters),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=pool_size_2),
            nn.Dropout(dropout),

            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_features)
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