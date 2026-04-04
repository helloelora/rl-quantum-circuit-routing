"""Dueling CNN network for D3QN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Number of observation channels from the environment
NUM_STATE_CHANNELS = 5


class DuelingCNN(nn.Module):
    """
    Dueling network architecture with CNN feature extractor.

    Input:  (batch, 5, N, N) state tensor
    Output: (batch, num_actions) Q-values

    Architecture:
        Conv2d(5->32, 3x3, same) -> ReLU
        Conv2d(32->64, 3x3, same) -> ReLU
        Conv2d(64->32, 3x3, same) -> ReLU
        Flatten -> 32*N*N
        V stream: Linear(->256) -> ReLU -> Linear(->1)
        A stream: Linear(->256) -> ReLU -> Linear(->num_actions)
        Q = V + A - mean(A)
    """

    def __init__(self, matrix_size: int, num_actions: int,
                 conv_channels: list = None, dueling_hidden: int = 256):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 32]

        layers = []
        in_ch = NUM_STATE_CHANNELS
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        flat_size = conv_channels[-1] * matrix_size * matrix_size

        self.value_stream = nn.Sequential(
            nn.Linear(flat_size, dueling_hidden),
            nn.ReLU(),
            nn.Linear(dueling_hidden, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(flat_size, dueling_hidden),
            nn.ReLU(),
            nn.Linear(dueling_hidden, num_actions),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 5, N, N) float32 tensor
        Returns:
            (batch, num_actions) Q-values
        """
        feat = self.features(x)
        feat = feat.flatten(start_dim=1)

        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


class ResBlock(nn.Module):
    """Residual block: two conv layers with a skip connection.

    If in_channels != out_channels, a 1x1 conv projects the skip path.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv to match channels on the skip path when they differ
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class DeepDuelingCNN(nn.Module):
    """Deep CNN with residual blocks, BatchNorm, and MaxPool.

    Moves model capacity from FC layers into the CNN feature extractor.
    Original DuelingCNN: 38K CNN (0.3%) + 12M FC (99.7%) = 12M total.
    This network:        ~492K CNN (62%) + ~301K FC (38%) = ~793K total.

    Architecture (default block_channels=[64, 128, 64]):
        Block 1: Conv(5->64,3x3) -> BN -> ReLU -> Conv(64->64,3x3) -> BN -> +skip -> ReLU -> MaxPool(2)
                 (batch,5,27,27) -> (batch,64,13,13)
        Block 2: Conv(64->128,3x3) -> BN -> ReLU -> Conv(128->128,3x3) -> BN -> +skip -> ReLU -> MaxPool(2)
                 (batch,64,13,13) -> (batch,128,6,6)
        Block 3: Conv(128->64,3x3) -> BN -> ReLU -> MaxPool(2)
                 (batch,128,6,6) -> (batch,64,3,3)
        AdaptiveAvgPool(3,3) -> Flatten -> 576
        Dueling V/A streams: Linear(576->hidden)->ReLU->Linear(->1 or ->num_actions)
        Q = V + A - mean(A)

    Input:  (batch, 5, N, N) — works for any N (topology-independent feature extraction)
    Output: (batch, num_actions) Q-values
    """

    def __init__(self, matrix_size: int, num_actions: int,
                 block_channels: list = None, dueling_hidden: int = 256,
                 pool_output_size: int = 3):
        super().__init__()
        if block_channels is None:
            block_channels = [64, 128, 64]

        # Block 1: ResBlock(5 -> ch0) + MaxPool
        self.block1 = ResBlock(NUM_STATE_CHANNELS, block_channels[0])
        self.pool1 = nn.MaxPool2d(2)

        # Block 2: ResBlock(ch0 -> ch1) + MaxPool
        self.block2 = ResBlock(block_channels[0], block_channels[1])
        self.pool2 = nn.MaxPool2d(2)

        # Block 3: single conv + BN (no skip — channels shrink) + MaxPool
        self.conv3 = nn.Conv2d(block_channels[1], block_channels[2], 3, padding=1)
        self.bn3 = nn.BatchNorm2d(block_channels[2])
        self.pool3 = nn.MaxPool2d(2)

        # Adaptive pool forces output to (pool_output_size x pool_output_size)
        # regardless of input spatial size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_output_size, pool_output_size))

        flat_size = block_channels[2] * pool_output_size * pool_output_size

        self.value_stream = nn.Sequential(
            nn.Linear(flat_size, dueling_hidden),
            nn.ReLU(),
            nn.Linear(dueling_hidden, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(flat_size, dueling_hidden),
            nn.ReLU(),
            nn.Linear(dueling_hidden, num_actions),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 5, N, N) float32 tensor
        Returns:
            (batch, num_actions) Q-values
        """
        # Block 1: (batch, 5, N, N) -> (batch, 64, N//2, N//2)
        x = self.pool1(self.block1(x))

        # Block 2: -> (batch, 128, N//4, N//4)
        x = self.pool2(self.block2(x))

        # Block 3: -> (batch, 64, N//8, N//8)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Adaptive pool: -> (batch, 64, 3, 3) regardless of input size
        x = self.adaptive_pool(x)

        # Flatten: -> (batch, 576)
        x = x.flatten(start_dim=1)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
