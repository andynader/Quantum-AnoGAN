from Generator import *
from Critic import *
from Training import *

X_normal = scinp.random.uniform(low=-1, high=1, size=(100, 4))
X_anomalous = scinp.random.uniform(low=50, high=100, size=(100, 4))

latent_dim = 2
num_layers = 2
upscaling_dimension = 4
batch_size = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

quantum_gen = QuantumGenerator(latent_dim, num_layers, upscaling_dimension, device)
quantum_gen = quantum_gen.to(device)

samples = [quantum_gen, quantum_gen]

critic = Critic(data_dimension=upscaling_dimension)
critic = critic.to(device)

# X_minibatch, z_minibatch, epsilons = sample_arrays(X_normal, quantum_gen, batch_size,
#                                                   upscaling_dimension, device)
#
# X_hat_minibatch = get_X_hat(X_minibatch, z_minibatch, epsilons)
#
# mean_L_C = critic_loss(X_minibatch, X_hat_minibatch, z_minibatch, critic, device)
# print(mean_L_C)

train_quantum_anogan(X_normal, quantum_gen, critic, device, n_iter=10, batch_size=64, n_critic=5)
