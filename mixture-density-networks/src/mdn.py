import math

import torch
import torch.nn as nn

from typing import List


class MixtureDensityNetwork(nn.Module):
    
    def __init__(self, layers: List[int], num_gaussians: int, output_dim: int):
        super().__init__()
        self.layers = layers
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim
        
        self.feature_extractor = nn.Sequential(*[
            nn.Sequential(nn.Linear(in_features, out_features), nn.Tanh())
            for in_features, out_features in zip(self.layers[:-1], self.layers[1:])
        ])
        
        self.mean_network   = nn.Linear(self.layers[-1], self.num_gaussians * self.output_dim)
        self.logvar_network = nn.Linear(self.layers[-1], self.num_gaussians)  # Common variance for all dimensions
        self.weight_network = nn.Sequential(
            nn.Linear(self.layers[-1], self.num_gaussians),
            nn.Softmax(-1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        means = self.mean_network(features).reshape(-1, self.num_gaussians, self.output_dim)
        logvars = self.logvar_network(features).reshape(-1, self.num_gaussians)
        weights = self.weight_network(features).reshape(-1, self.num_gaussians)
        variances = logvars.exp()

        return means, variances, weights
    
    def loss_func(self, targs, means, variances, weights):
        # Note: we can't use an analytic expression for the log likelihood
        # since we need to sum the mixture densities _inside_ the logarithm
        reshaped_targs = targs.unsqueeze(-2).expand_as(means)
        separate_densities = self.gaussian_pdf(reshaped_targs, means, variances)
        mixed_densities = (separate_densities * weights).sum(-1)
        losses = -mixed_densities.log()
        loss = losses.mean()
        return loss
    
    def gaussian_pdf(self, x, mu, variance):
        coef_term = torch.reciprocal(2 * math.pi * variance).pow(self.output_dim / 2)
        exp_term = torch.exp(-0.5 * (x - mu).pow(2).sum(-1) * torch.reciprocal(variance))
        return coef_term * exp_term
