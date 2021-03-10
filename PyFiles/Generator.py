import pennylane as qml
from pennylane import numpy as np
import numpy as scinp

import torch
from torch.nn.init import xavier_uniform_
from torch import nn


def state_preparation(latent_variables):
    for i in range(len(latent_variables)):
        qml.RX(latent_variables[i], wires=i)


def layer(W):
    num_wires = len(W)
    for i in range(num_wires):
        random_rot = np.random.choice([qml.RX, qml.RY, qml.RZ])
        random_rot(W[i], wires=i)

    for i in range(num_wires - 1):
        qml.CNOT(wires=[i, i + 1])


def variational_circuit(latent_variables, weights):
    # The number of wires is exactly equal to
    # the dimensionality of the latent variables.

    state_preparation(latent_variables)

    for W in weights:
        layer(W)

    return [qml.expval(qml.PauliZ(i)) for i in range(len(latent_variables))]


class QuantumGenerator(nn.Module):

    def __init__(self, latent_dim, num_layers, upscaling_dimension, device):
        super(QuantumGenerator, self).__init__()

        # We convert the variational quantum circuit to a pytorch qnode.

        quantum_dev = qml.device("qulacs.simulator", wires=latent_dim, gpu=True)
        variational_circuit_torch = qml.QNode(variational_circuit, quantum_dev, interface="torch")
        # We store the quantum classifier
        self.vqc = variational_circuit_torch

        self.device = device
        self.latent_dim = latent_dim
        # We initalize and store the quantum classifier's weights
        W = torch.Tensor(num_layers, latent_dim).uniform_(-np.pi, np.pi).to(self.device)

        # We specify that the quantum classifier weights parameters of the
        # hybrid quantum classical generator, and thus should be differentiated.

        self.quantum_weights = nn.Parameter(W)

        # We define the upscaling layer, and we initialize it using the
        # glorot uniform weight initialization
        self.upscaling_layer = nn.Linear(latent_dim, upscaling_dimension)
        xavier_uniform_(self.upscaling_layer.weight)

    def forward(self):
        # We define the latent variables, and pass them through a quantum generator.
        latent_variables = torch.Tensor(self.latent_dim).uniform_(-np.pi, np.pi).to(self.device)
        quantum_out = self.vqc(latent_variables, self.quantum_weights).float().to(self.device)
        generated_sample = self.upscaling_layer(quantum_out)
        return generated_sample
