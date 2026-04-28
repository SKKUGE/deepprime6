from torch import nn


class ParallelDeepPrimeModels(nn.Module):
    """A module that applies a list of models in parallel to the input.

    Args:
        models (list): A list of models to apply.

    Attributes:
        models (nn.ModuleList): A module list containing the models.
    """

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, g, x):
        """Forward pass of the parallel model.

        Args:
            g (tensor): The input genetic features.
            x (tensor): The input biofeaure tensor.

        Returns:
            list: A list of outputs from each model.
        """
        return [model(g, x) for model in self.models]
