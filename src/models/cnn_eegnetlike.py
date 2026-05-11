import torch
import torch.nn as nn


class EEGNetLike(nn.Module):
    """
    EEGNet-like CNN for cleaned/raw EEG data.

    Expected input: (batch_size, 1, n_channels, n_timepoints)

    Example: (batch_size, 1, 32, 900)
    """

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        dropout: float = 0.5,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        temporal_kernel_size: int = 64,
        separable_kernel_size: int = 16,
    ):
        super().__init__()

        F1 = temporal_filters
        D = depth_multiplier
        F2 = F1 * D

        self.feature_extractor = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, temporal_kernel_size),
                padding=(0, temporal_kernel_size // 2),
                bias=False
            ),
            nn.BatchNorm2d(F1),

            # Spatial / depthwise convolution across EEG channels
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),

            # Separable-like convolution
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, separable_kernel_size),
                padding=(0, separable_kernel_size // 2),
                groups=F1 * D,
                bias=False
            ),
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),

            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            flattened_size = self.feature_extractor(dummy).shape[1]

        self.classifier = nn.Linear(flattened_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        logits = self.classifier(x)

        return logits.squeeze(1)