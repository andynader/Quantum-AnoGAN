import torch
from torch import nn


class ClassicalGenerator(nn.Module):

    def __init__(self, latent_dim, hidden_layer_sizes, data_dimension, device):
        super(ClassicalGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.data_dimension = data_dimension
        self.device = device

        # hidden_layer_sizes is of the form [h1_size,h2_size,h3_size,...]
        # where hi_size is the size of the ith hidden layer

        layer_sizes = [latent_dim] + hidden_layer_sizes + [data_dimension]
        layers = []
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            layers.append(nn.Linear(in_size, out_size))
            if i != len(layer_sizes) - 2:
                #                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)

    def forward(self):
        z = torch.normal(mean=0.0, std=2.0, size=(self.latent_dim,)).to(self.device)
        generated_sample = self.network(z)
        return generated_sample
