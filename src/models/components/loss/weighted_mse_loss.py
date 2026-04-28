# Got implementation from here: https://github.com/yumin-c/DeepPrime/tree/b97a1b77375bbdb56f828829e65562315d832616
import torch
import torch.nn as nn

LABEL_IDX = -3
SUB_IDX = -3
INS_IDX = -2
DEL_IDX = -1


class ImprovedBalancedLoss(nn.Module):
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        beta=0.9999,
        substitution_weight=1.0,
        insertion_weight=0.7,
        deletion_weight=0.6,
    ):
        """Initializes the WeightedMSELoss object with the specified parameters.

        Args:
            alpha (float): The alpha parameter for Focal Loss (Lin et al., 2018). Default is 0.25.
            gamma (float): The gamma parameter for Focal Loss (Lin et al., 2018). Default is 2.0.
            beta (float): The beta parameter for Class-Balanced Loss (Cui et al., 2019). Default is 0.9999.
            substitution_weight (float): The weight for substitution errors. Default is 1.0.
            insertion_weight (float): The weight for insertion errors. Default is 0.7.
            deletion_weight (float): The weight for deletion errors. Default is 0.6.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.substitution_weight = substitution_weight
        self.insertion_weight = insertion_weight
        self.deletion_weight = deletion_weight

    def forward(self, pred, actual):
        # Apply log1p transformation to handle skewed distributions
        y = torch.log1p(actual[:, :LABEL_IDX])
        pred = torch.log1p(pred)

        # Calculate focal loss component
        # Focal Loss (Lin et al., 2018)
        pt = torch.exp(-torch.abs(y - pred))
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * (y - pred) ** 2

        # Calculate balanced component
        # Class-Balanced Loss (Cui et al., 2019)
        effective_num = 1.0 - torch.pow(self.beta, torch.exp(y))
        weights = (1.0 - self.beta) / effective_num
        weights = weights / torch.sum(weights) * len(weights)  # Normalize weights by the average

        # Combine focal and balanced components
        loss = weights * focal_loss

        # Apply different weights for edit types
        edit_weights = torch.ones_like(actual[:, :LABEL_IDX])
        edit_weights[actual[:, SUB_IDX] == 1] = self.substitution_weight  # Substitution
        edit_weights[actual[:, INS_IDX] == 1] = self.insertion_weight  # Insertion
        edit_weights[actual[:, DEL_IDX] == 1] = self.deletion_weight  # Deletion

        weighted_loss = loss * edit_weights

        return torch.mean(weighted_loss)


class BalancedMSELoss(nn.Module):
    """Custom loss function that applies a weighted mean squared error (MSE) loss. The loss is
    calculated separately for each target class and then summed up. The weights for each class can
    be adjusted using the `factor` attribute.

    Args:
        scale (bool, optional): Whether to apply scaling to the MSE loss using `ScaledMSELoss`.
            If False, the regular `nn.MSELoss` is used. Default is True.
    """

    def __init__(
        self,
        scale=True,
        scale_threshold=3.0,
        minimum_scaling_factor=5.0,
        alpha=6.0,
        substitution_weight=1.0,
        insertion_weight=0.7,
        deletion_weight=0.6,
    ):
        super().__init__()

        self.substitution_weight: float = substitution_weight
        self.insertion_weight: float = insertion_weight
        self.deletion_weight: float = deletion_weight

        if scale:
            self.mse = ScaledMSELoss(
                alpha=alpha,
                scale_threshold=scale_threshold,
                minimum_scaling_factor=minimum_scaling_factor,
            )
            print("Applying ScaledMSELoss")
        else:
            self.mse = nn.MSELoss()
            print("Applying MSELoss without scaling")

    def forward(self, pred, actual):
        # Apply log1p transformation to handle skewed distributions
        y = torch.log1p(actual[:, :LABEL_IDX])
        pred = torch.log1p(pred)

        # Weighting loss by their edit types
        l1 = (
            self.mse(pred[actual[:, SUB_IDX] == 1], y[actual[:, SUB_IDX] == 1])
            * self.substitution_weight
        )  # Sub
        l2 = (
            self.mse(pred[actual[:, INS_IDX] == 1], y[actual[:, INS_IDX] == 1])
            * self.insertion_weight
        )  # Ins
        l3 = (
            self.mse(pred[actual[:, DEL_IDX] == 1], y[actual[:, DEL_IDX] == 1])
            * self.deletion_weight
        )  # Del

        return l1 + l2 + l3


class ScaledMSELoss(nn.Module):
    def __init__(
        self, alpha: float = 6.0, scale_threshold: float = 3.0, minimum_scaling_factor: float = 5.0
    ):
        super().__init__()
        self.alpha: float = alpha
        self.scale_threshold: float = scale_threshold
        self.minimum_scaling_factor: float = minimum_scaling_factor

    def forward(self, pred, y):
        # Create a mask for non-NaN values
        valid_mask = ~torch.isnan(pred) & ~torch.isnan(y)

        # Filter out NaN values
        pred_valid = pred[valid_mask]
        y_valid = y[valid_mask]

        # Compute mu for valid values
        mu = torch.minimum(
            torch.exp(self.alpha * (y_valid - self.scale_threshold)) + 1,
            torch.ones_like(y_valid) * self.minimum_scaling_factor,
        )

        # Compute the loss on valid values
        squared_diff = (y_valid - pred_valid) ** 2
        weighted_squared_diff = mu * squared_diff
        loss = torch.mean(weighted_squared_diff)

        return loss
