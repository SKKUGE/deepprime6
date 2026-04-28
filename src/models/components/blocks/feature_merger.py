import torch
import torch.nn as nn

from src.models.components.blocks.residual_layer import ResidualCNNBlock


class ResidualFeatureMerger(nn.Module):
    def __init__(
        self,
        num_residual_blocks: int,
        num_conv_layers_per_block: int,
        layer_in_in_channels: int,
        layer_hidden_in_channels: int,
        adaptive_avgpool_output_size: int,
        initial_conv_kernel_size: int = 7,
        initial_conv_maxpool: bool = True,
        initial_conv_maxpool_kernel_size: int = 3,
        final_avgpool: bool = True,
    ):
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv1d(
                layer_in_in_channels,
                layer_hidden_in_channels,
                kernel_size=initial_conv_kernel_size,
                stride=1,
                padding=initial_conv_kernel_size // 2,
            ),
            nn.BatchNorm1d(layer_hidden_in_channels),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=initial_conv_maxpool_kernel_size,
                stride=1,
                padding=(initial_conv_maxpool_kernel_size - 1) // 2,
            )
            if initial_conv_maxpool
            else nn.Identity(),
        )

        self.residual_blocks = nn.Sequential()
        in_channels = layer_hidden_in_channels
        for i in range(num_residual_blocks):
            out_channels = in_channels * 2 if i % 2 == 0 and i > 0 else in_channels
            self.residual_blocks.add_module(
                f"residual_block_{i}",
                ResidualCNNBlock(in_channels, out_channels, num_conv_layers_per_block),
            )
            in_channels = out_channels

        self.avgpool = (
            nn.AdaptiveAvgPool1d(adaptive_avgpool_output_size) if final_avgpool else nn.Identity()
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.residual_blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
