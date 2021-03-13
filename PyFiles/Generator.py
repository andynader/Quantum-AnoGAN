import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import StronglyEntanglingLayers
from pennylane.init import strong_ent_layers_normal
import numpy as scinp

import torch
from torch.nn.init import xavier_uniform_
from torch import nn


def state_preparation(latent_variables):
    for i in range(len(latent_variables)):
        qml.RX(latent_variables[i], wires=i)


def generate_rotation_bases(num_layers, num_wires):
    rotation_bases_all_layers = []
    for i in range(num_layers):
        rotation_bases_single_layer = [np.random.choice([qml.RX, qml.RY, qml.RZ]) for j in range(num_wires)]
        rotation_bases_all_layers.append(rotation_bases_single_layer)
    return rotation_bases_all_layers


def layer(W, rot_bases):
    num_wires = len(W)
    for i in range(num_wires):
        rot = rot_bases[i]
        rot(W[i], wires=i)

    for i in range(num_wires - 1):
        qml.CNOT(wires=[i, i + 1])


def variational_circuit(latent_variables, weights, rotation_bases_all_layers=None):
    # The number of wires is exactly equal to
    # the dimensionality of the latent variables.

    state_preparation(latent_variables)

    for i in range(len(weights)):
        layer(weights[i], rotation_bases_all_layers[i])

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

        # We generate the rotation bases
        self.rotation_bases_all_layers = generate_rotation_bases(num_layers=num_layers, num_wires=latent_dim)

        self.upscaling_dimension = upscaling_dimension
        # We initialize and store the quantum classifier's weights
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
        quantum_out = self.vqc(latent_variables, self.quantum_weights, self.rotation_bases_all_layers).float().to(
            self.device)
        generated_sample = self.upscaling_layer(quantum_out)
        return generated_sample
