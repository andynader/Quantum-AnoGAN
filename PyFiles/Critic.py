import torch
from torch import nn
from torch.nn.init import xavier_uniform_
import numpy as scinp


class Critic(nn.Module):

    def __init__(self, data_dimension, hidden_layer_sizes):
        super(Critic, self).__init__()

        layer_sizes = [data_dimension] + hidden_layer_sizes + [1]

        layers = []

        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            layers.append(nn.Linear(in_size, out_size))
            if i != len(layer_sizes) - 2:
                #                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        critic_output = self.network(x)
        return critic_output
