import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSoftmax(nn.Module):
    """Gumbel-Softmax sampling for categorical distributions"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, logprobs: torch.Tensor, temperature: torch.Tensor):
        rsample = (logprobs + self.gumbel_sample(logprobs.shape, device=logprobs.device)) / temperature
        gumbel_softmax = rsample.softmax(-1)
        argmax = F.one_hot(gumbel_softmax.argmax(-1), num_classes=gumbel_softmax.size(-1))
        return argmax
    
    @staticmethod
    def gumbel_sample(shape: torch.Size, device: torch.device) -> torch.Tensor:
        unif_sample = torch.rand(shape, device=device)
        return -torch.log(-torch.log(unif_sample))
