import torch
import torch.nn as nn


class CNNHybrid(nn.Module):
    """
    Hybrid CNN model with two inputs:

    Input 1: Raw EEG: (batch_size, 1, n_channels, n_timepoints)
    Input 2: EEG features: (batch_size, n_channels, n_features)
    Output: logits: (batch_size, n_classes)
    Use with: nn.CrossEntropyLoss()
    """

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        n_features: int,
        n_classes: int = 2,

        raw_temporal_filters: int = 16,
        raw_kernel_size: int = 64,
        raw_pool_size: int = 8,

        feature_conv1_filters: int = 32,
        feature_conv2_filters: int = 64,
        feature_kernel_size_1: int = 3,
        feature_kernel_size_2: int = 3,

        raw_embedding_units: int = 64,
        feature_embedding_units: int = 64,
        fusion_hidden_units: int = 64,

        dropout: float = 0.5,
    ):
        super().__init__()

        self.raw_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=raw_temporal_filters,
                kernel_size=(1, raw_kernel_size),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(raw_temporal_filters),
            nn.ELU(),

            nn.Conv2d(
                in_channels=raw_temporal_filters,
                out_channels=raw_temporal_filters,
                kernel_size=(n_channels, 1),
                groups=raw_temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(raw_temporal_filters),
            nn.ELU(),

            nn.AvgPool2d(kernel_size=(1, raw_pool_size)),
            nn.Dropout(dropout),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_raw = torch.zeros(1, 1, n_channels, n_timepoints)
            raw_flattened_size = self.raw_branch(dummy_raw).shape[1]

        self.raw_projection = nn.Sequential(
            nn.Linear(raw_flattened_size, raw_embedding_units),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.feature_branch = nn.Sequential(
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=feature_conv1_filters,
                kernel_size=feature_kernel_size_1,
                padding=feature_kernel_size_1 // 2,
                bias=False,
            ),
            nn.BatchNorm1d(feature_conv1_filters),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Conv1d(
                in_channels=feature_conv1_filters,
                out_channels=feature_conv2_filters,
                kernel_size=feature_kernel_size_2,
                padding=feature_kernel_size_2 // 2,
                bias=False,
            ),
            nn.BatchNorm1d(feature_conv2_filters),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_features = torch.zeros(1, n_channels, n_features)
            feature_flattened_size = self.feature_branch(dummy_features).shape[1]

        self.feature_projection = nn.Sequential(
            nn.Linear(feature_flattened_size, feature_embedding_units),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        fusion_input_size = raw_embedding_units + feature_embedding_units

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden_units),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_units, n_classes),
        )

    def forward(self, x_raw: torch.Tensor, x_features: torch.Tensor) -> torch.Tensor:
        raw_embedding = self.raw_branch(x_raw)
        raw_embedding = self.raw_projection(raw_embedding)

        feature_embedding = self.feature_branch(x_features)
        feature_embedding = self.feature_projection(feature_embedding)

        combined = torch.cat([raw_embedding, feature_embedding], dim=1)

        logits = self.classifier(combined)
        return logits
