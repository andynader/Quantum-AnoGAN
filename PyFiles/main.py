from Generator import *
from Critic import *

latent_dim = 2
num_layers = 2
upscaling_dimension = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

quantum_gen = QuantumGenerator(latent_dim, num_layers, upscaling_dimension, device)
quantum_gen = quantum_gen.to(device)

critic = Critic(data_dimension=upscaling_dimension)
critic = critic.to(device)

