import math

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

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

        normals = Normal(means, variances.unsqueeze(-1))
        weight_dist = Categorical(probs=weights)

        return normals, weight_dist
    
    def loss_func(self, targs: torch.Tensor, normals: Normal, weight_dist: Categorical):
        """Return the negative log-likelihood loss"""
        reshaped_targs = targs.unsqueeze(1).expand_as(normals.loc)

        # Since each output dim of the multivariate Gaussian is independent, we
        # find the log probability of the target value by summing the log
        # probabilities along each dimension. This is equivalent to multiplying
        # the likelihoods and then taking their logarithm)
        log_probs = normals.log_prob(reshaped_targs)
        log_probs = log_probs.sum(-1)
        log_probs += weight_dist.probs.log() # Equivalent to multiplying by weights then taking the log
        log_likelihoods = torch.logsumexp(log_probs, dim=1)
        nll = -log_likelihoods.mean()
        
        return nll
    
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass and then sample from the outputs distributions"""
        with torch.no_grad():
            normals, weight_dist = self(x)

        idxs = weight_dist.sample()
        idxs = idxs.unsqueeze(-1)
        idxs = torch.stack([idxs, idxs], dim=-1)
        preds = normals.loc.gather(1, idxs)

        return preds
