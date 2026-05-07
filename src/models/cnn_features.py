import torch
import torch.nn as nn


class CNNFeatures2D(nn.Module):
    def __init__(self, n_channels: int, n_features: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_features)
            flattened_size = self.model(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x