import torch.nn as nn


class FCResidualBlock(nn.Module):
    """Fully connected residual block.

    Args:
        input_dim (int): The dimensionality of the input.
        hidden_dim (int): The dimensionality of the hidden layer.
        output_dim (int): The dimensionality of the output.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        bn1 (nn.BatchNorm1d): Batch normalization layer for the hidden layer.
        bn2 (nn.BatchNorm1d): Batch normalization layer for the output layer.
        relu (nn.ReLU): ReLU activation function.
        shortcut (nn.Module): Shortcut connection to handle input and output dimension mismatch.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        # Shortcut connection
        self.shortcut = nn.Identity()
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """Forward pass of the FCResidualBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        residual = self.shortcut(x)

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_conv_layers: int):
        super().__init__()
        self.conv_layers = nn.ModuleList()

        for i in range(num_conv_layers):
            if i == 0:
                self.conv_layers.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
                )
            else:
                self.conv_layers.append(
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
                )

            if i < num_conv_layers - 1:
                self.conv_layers.append(nn.BatchNorm1d(out_channels))
                self.conv_layers.append(nn.ReLU())

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1), nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        for layer in self.conv_layers:
            x = layer(x)
        return x + self.shortcut(residual)
