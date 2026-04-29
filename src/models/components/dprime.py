import torch
from torch import nn
from torch.nn import functional as F

# [BUG] AttributeError: module 'src.models.components.dprime' has no attribute 'GeneInteractionModelOriginalImplementation'
# [2026-04-30 00:20:10,297][src.utils.utils][ERROR] - [rank: 0] module 'src.models.components.dprime' has no attribute 'GeneInteractionModelOriginalImplementation'
# TODO: Module dependency check is necessary

class GeneInteractionModelOriginalImplementation(nn.Module):
    """GeneInteractionModel is a PyTorch module that represents a gene interaction model from
    DeepPrime (Yu et al., 2023 Cell).

    The following module is adopted from the original implementation of DeepPrime/src/dprime.py.
    The original repo is: https://github.com/hkimlab/DeepPrime
    """

    def __init__(
        self,
        c1_in_channels: int = 4,
        c1_out_channels: int = 128,
        c1_kernel_size: tuple[int, int] = (2, 3),
        c1_stride: int = 1,
        c1_padding: tuple[int, int] = (0, 1),
        c1_batchnorm_features: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_features: int = 24,
        dropout: float = 0.1,
        c2_out_channels: int = 108,
        c2_kernel_size: int = 3,
        c2_stride: int = 1,
        c2_padding: int = 1,
        r_in_features: int = 128,
        s_out_features: int = 12,
        d_hidden1: int = 96,
        d_hidden2: int = 64,
        d_out_features: int = 128,
        head_in_features: int = 140,
        head_out_features: int = 1,
    ):
        """Initialize the GeneInteractionModel class.

        Args:
            c1_in_channels (int): Number of input channels for the first convolutional layer. Defaults to 4.
            c1_out_channels (int): Number of output channels for the first convolutional layer. Defaults to 128.
            c1_kernel_size (tuple[int, int]): Kernel size for the first convolutional layer. Defaults to (2, 3).
            c1_stride (int): Stride for the first convolutional layer. Defaults to 1.
            c1_padding (tuple[int, int]): Padding for the first convolutional layer. Defaults to (0, 1).
            c1_batchnorm_features (int): Number of features for batch normalization in the first convolutional layer. Defaults to 128.
            hidden_size (int): The number of features in the hidden state of the GRU. Defaults to 128.
            num_layers (int): The number of recurrent layers in the GRU. Defaults to 1.
            num_features (int): The number of input features. Defaults to 24.
            dropout (float): The dropout probability. Defaults to 0.1.
            c2_in_channels (int): Number of input channels for the second convolutional layer. Defaults to 128.
            c2_out_channels (int): Number of output channels for the second convolutional layer. Defaults to 108.
            c2_kernel_size (int): Kernel size for the second convolutional layer. Defaults to 3.
            c2_stride (int): Stride for the second convolutional layer. Defaults to 1.
            c2_padding (int): Padding for the second convolutional layer. Defaults to 1.
            r_in_features (int): Number of input features for the GRU layer. Defaults to 128.
            s_in_features (int): Number of input features for the linear layer after the GRU layer. Defaults to 2 * hidden_size.
            s_out_features (int): Number of output features for the linear layer after the GRU layer. Defaults to 12.
            d_in_features (int): Number of input features for the linear layers in the d module. Defaults to num_features.
            d_hidden1 (int): Number of hidden units for the first linear layer in the d module. Defaults to 96.
            d_hidden2 (int): Number of hidden units for the second linear layer in the d module. Defaults to 64.
            d_out_features (int): Number of output features for the last linear layer in the d module. Defaults to 128.
            head_in_features (int): Number of input features for the linear layers in the head module. Defaults to 140.
            head_out_features (int): Number of output features for the last linear layer in the head module. Defaults to 1.
        """

        super().__init__()

        # Calculating dynamic values
        s_in_features: int = 2 * hidden_size
        d_in_features: int = num_features

        self.c1 = nn.Sequential(
            nn.Conv2d(
                in_channels=c1_in_channels,
                out_channels=c1_out_channels,
                kernel_size=c1_kernel_size,
                stride=c1_stride,
                padding=c1_padding,
            ),
            nn.BatchNorm2d(c1_batchnorm_features),
            nn.GELU(),
        )
        self.c2 = nn.Sequential(
            nn.Conv1d(
                in_channels=c1_out_channels,
                out_channels=c2_out_channels,
                kernel_size=c2_kernel_size,
                stride=c2_stride,
                padding=c2_padding,
            ),
            nn.BatchNorm1d(c2_out_channels),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=c2_out_channels,
                out_channels=c2_out_channels,
                kernel_size=c2_kernel_size,
                stride=c2_stride,
                padding=c2_padding,
            ),
            nn.BatchNorm1d(c2_out_channels),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=c2_out_channels,
                out_channels=c1_out_channels,
                kernel_size=c2_kernel_size,
                stride=c2_stride,
                padding=c2_padding,
            ),
            nn.BatchNorm1d(c1_out_channels),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        self.r = nn.GRU(
            r_in_features, hidden_size, num_layers, batch_first=True, bidirectional=True
        )

        self.s = nn.Linear(s_in_features, s_out_features, bias=False)

        self.d = nn.Sequential(
            nn.Linear(d_in_features, d_hidden1, bias=False),
            nn.BatchNorm1d(d_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden1, d_hidden2, bias=False),
            nn.BatchNorm1d(d_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden2, d_out_features, bias=False),
            nn.BatchNorm1d(d_out_features),
            nn.ReLU(),
        )  # TODO: ablation study

        self.head = nn.Sequential(
            nn.BatchNorm1d(head_in_features),
            nn.Dropout(dropout),
            nn.Linear(head_in_features, head_out_features, bias=True),
        )

    def forward(self, g, x):
        """Forward pass of the DPrime model.

        Args:
            g (torch.Tensor): Input tensor representing g (pegRNA sequences).
            x (torch.Tensor): Input tensor representing x (biofeatures).

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """

        g = torch.squeeze(self.c1(g), 2)
        g = self.c2(g)
        g, _ = self.r(torch.transpose(g, 1, 2))
        g = self.s(g[:, -1, :])

        x = self.d(x)

        out = self.head(torch.cat((g, x), dim=1))

        return F.softplus(out)



class GeneInteractionModelVanilla(nn.Module):
    """GeneInteractionModelVanilla is a PyTorch module that perfectly replicates the structural 
    definition of the gene interaction model from DeepPrime (Yu et al., 2023 Cell)
    as it is instantiated in genet package, specifically without the final BN and ReLU layers
    in the `d` module, and without the BN layer in the `head` module.
    """

    def __init__(
        self,
        c1_in_channels: int = 4,
        c1_out_channels: int = 128,
        c1_kernel_size: tuple[int, int] = (2, 3),
        c1_stride: int = 1,
        c1_padding: tuple[int, int] = (0, 1),
        c1_batchnorm_features: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_features: int = 24,
        dropout: float = 0.1,
        c2_out_channels: int = 108,
        c2_kernel_size: int = 3,
        c2_stride: int = 1,
        c2_padding: int = 1,
        r_in_features: int = 128,
        s_out_features: int = 12,
        d_hidden1: int = 96,
        d_hidden2: int = 64,
        d_out_features: int = 128,
        head_in_features: int = 140,
        head_out_features: int = 1,
    ):
        super().__init__()

        # Calculating dynamic values
        s_in_features: int = 2 * hidden_size
        d_in_features: int = num_features

        self.c1 = nn.Sequential(
            nn.Conv2d(
                in_channels=c1_in_channels,
                out_channels=c1_out_channels,
                kernel_size=c1_kernel_size,
                stride=c1_stride,
                padding=c1_padding,
            ),
            nn.BatchNorm2d(c1_batchnorm_features),
            nn.GELU(),
        )
        self.c2 = nn.Sequential(
            nn.Conv1d(
                in_channels=c1_out_channels,
                out_channels=c2_out_channels,
                kernel_size=c2_kernel_size,
                stride=c2_stride,
                padding=c2_padding,
            ),
            nn.BatchNorm1d(c2_out_channels),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=c2_out_channels,
                out_channels=c2_out_channels,
                kernel_size=c2_kernel_size,
                stride=c2_stride,
                padding=c2_padding,
            ),
            nn.BatchNorm1d(c2_out_channels),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=c2_out_channels,
                out_channels=c1_out_channels,
                kernel_size=c2_kernel_size,
                stride=c2_stride,
                padding=c2_padding,
            ),
            nn.BatchNorm1d(c1_out_channels),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        self.r = nn.GRU(
            r_in_features, hidden_size, num_layers, batch_first=True, bidirectional=True
        )

        self.s = nn.Linear(s_in_features, s_out_features, bias=False)

        # Vanilla: no final BN, no final ReLU
        self.d = nn.Sequential(
            nn.Linear(d_in_features, d_hidden1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden1, d_hidden2, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden2, d_out_features, bias=False),
        ) 

        # Vanilla: head matches Genet's original
        self.head = nn.Sequential(
            nn.BatchNorm1d(head_in_features),
            nn.Dropout(dropout),
            nn.Linear(head_in_features, head_out_features, bias=True),
        )

    def forward(self, g, x):
        g = torch.squeeze(self.c1(g), 2)
        g = self.c2(g)
        g, _ = self.r(torch.transpose(g, 1, 2))
        g = self.s(g[:, -1, :])

        x = self.d(x)

        out = self.head(torch.cat((g, x), dim=1))

        return F.softplus(out)



if __name__ == "__main__":
    _ = GeneInteractionModelVanilla()
    print("GeneInteractionModelVanilla is successfully initialized.")