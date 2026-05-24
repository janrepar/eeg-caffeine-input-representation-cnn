import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetLike(nn.Module):
    """
    EEGNet-like CNN for raw/cleaned EEG data.

    Architecture based on EEGNet:
        Input: (batch, 1, C, T)

        Block 1:
            Conv2D(F1, kernel=(1, temporal_kernel_size), same)
            BatchNorm
            DepthwiseConv2D(F1 * D, kernel=(C, 1), valid, groups=F1)
            BatchNorm
            ELU
            AveragePool2D(1, 4)
            Dropout

        Block 2:
            SeparableConv2D:
                depthwise temporal Conv2D(kernel=(1, separable_kernel_size), same, groups=F1*D)
                pointwise Conv2D(kernel=(1, 1), out_channels=F2)
            BatchNorm
            ELU
            AveragePool2D(1, 8)
            Dropout

        Classifier:
            Flatten
            Dense n_classes

    Expected input:
        (batch_size, 1, n_channels, n_timepoints)

    Output:
        logits with shape (batch_size, n_classes)

    Use with:
        nn.CrossEntropyLoss()
    """

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        n_classes: int = 2,
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
            # ---------------------------------------------------------
            # Block 1: temporal convolution
            # Output: (batch, F1, C, T)
            # ---------------------------------------------------------
            Conv2dSame(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, temporal_kernel_size),
                bias=False,
            ),
            nn.BatchNorm2d(F1),

            # ---------------------------------------------------------
            # Block 1: depthwise spatial convolution across channels
            # Output: (batch, F1 * D, 1, T)
            # ---------------------------------------------------------
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),

            # ---------------------------------------------------------
            # Block 2: separable convolution
            # depthwise temporal convolution
            # ---------------------------------------------------------
            Conv2dSame(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, separable_kernel_size),
                groups=F1 * D,
                bias=False,
            ),

            # pointwise convolution
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),

            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            flattened_size = self.feature_extractor(dummy).shape[1]

        self.classifier = nn.Linear(flattened_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        logits = self.classifier(x)
        return logits


class Conv2dSame(nn.Module):
    """
    Conv2D with TensorFlow/Keras-like 'same' padding. EEGNet descriptions often assume mode='same'.
    """

    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=False):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        kh, kw = self.kernel_size

        pad_h_total = kh - 1
        pad_w_total = kw - 1

        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top

        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)