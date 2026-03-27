"""Dueling CNN network for D3QN."""

import torch
import torch.nn as nn


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
