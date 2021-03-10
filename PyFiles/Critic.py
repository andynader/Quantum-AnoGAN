import torch
from torch import nn
from torch.nn.init import xavier_uniform_
import numpy as scinp


class Critic(nn.Module):

    def __init__(self, data_dimension, hidden_dimensionality=None):
        super(Critic, self).__init__()

        if hidden_dimensionality is None:
            hidden_dimensionality = [16, 8]

        self.flatten = nn.Flatten()
        self.layer_dimensions = [data_dimension] + hidden_dimensionality + [1]

        layers = []

        for i in range(len(self.layer_dimensions) - 1):
            in_dim = self.layer_dimensions[i]
            out_dim = self.layer_dimensions[i + 1]
            linear_layer = nn.Linear(in_dim, out_dim)
            xavier_uniform_(linear_layer.weight)
            layers.append(linear_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        critic_output = self.network(x)
        return critic_output
