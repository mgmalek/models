import torch.nn as nn


class MLP(nn.Sequential):
    """A simple parameterised MLP"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        units_per_layer: int,
        hidden_layers: int,
        nonlinearity: nn.Module = nn.ReLU,
        output_nonlinearity: bool = False
    ):
        features = [input_dim] + [units_per_layer] * hidden_layers
        super().__init__(
            # Hidden Layers
            *[
                nn.Sequential(nn.Linear(in_feats, out_feats), nonlinearity())
                for in_feats, out_feats in zip(features[:-1], features[1:])
            ],

            # Output Layer
            nn.Linear(units_per_layer, output_dim),
            nonlinearity if output_nonlinearity else nn.Identity(),
        )
