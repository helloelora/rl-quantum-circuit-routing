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
    """Residual block with LayerNorm (via GroupNorm(1)).

    Uses GroupNorm with 1 group instead of BatchNorm. This is mathematically
    equivalent to LayerNorm over (C,H,W) but has no train/eval mode
    difference — critical for DQN where the online net (train mode) and
    target net (eval mode) must produce consistent Q-values.

    BatchNorm caused Q-value divergence in V7 runs 032/033/035 because
    the online and target networks used different normalization statistics.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.ln1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.ln2 = nn.GroupNorm(1, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.GroupNorm(1, out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class DeepDuelingCNN(nn.Module):
    """Deep residual CNN with LayerNorm, no pooling.

    V8 fix for V7 failures (runs 032/033/035 all diverged). Two changes:
      1. LayerNorm (GroupNorm(1)) replaces BatchNorm — no train/eval mismatch
      2. No pooling — full 27x27 spatial resolution preserved so the
         network can distinguish all 20 SWAP edges precisely

    Architecture (default block_channels=[64, 128, 64]):
        Block 1: ResBlock(5->64)   Conv->LN->ReLU->Conv->LN->+skip->ReLU  (27x27)
        Block 2: ResBlock(64->128) Conv->LN->ReLU->Conv->LN->+skip->ReLU  (27x27)
        Compress: Conv 1x1 (128->32) -> ReLU                               (27x27)
        Flatten -> 32*27*27 = 23,328
        V stream: Linear(23328->hidden)->ReLU->Linear(->1)
        A stream: Linear(23328->hidden)->ReLU->Linear(->num_actions)
        Q = V + A - mean(A)

    Same flat_size as the original DuelingCNN but with 7 conv layers
    (including 1x1) and skip connections for deeper feature extraction.

    Input:  (batch, 5, N, N)
    Output: (batch, num_actions) Q-values
    """

    def __init__(self, matrix_size: int, num_actions: int,
                 block_channels: list = None, dueling_hidden: int = 256,
                 pool_output_size: int = 3):
        super().__init__()
        if block_channels is None:
            block_channels = [64, 128, 64]

        # pool_output_size kept in signature for config compat but unused

        self.block1 = ResBlock(NUM_STATE_CHANNELS, block_channels[0])
        self.block2 = ResBlock(block_channels[0], block_channels[1])

        # 1x1 conv compresses channels, not spatial dims
        self.compress = nn.Sequential(
            nn.Conv2d(block_channels[1], 32, kernel_size=1),
            nn.ReLU(),
        )

        flat_size = 32 * matrix_size * matrix_size

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
        x = self.block1(x)       # (batch, 64, 27, 27)
        x = self.block2(x)       # (batch, 128, 27, 27)
        x = self.compress(x)     # (batch, 32, 27, 27)
        x = x.flatten(start_dim=1)  # (batch, 23328)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
