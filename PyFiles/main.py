from Generator import *
from Critic import *
from Training import *

X_normal = scinp.random.uniform(low=-10, high=-5, size=(100, 4))
X_anomalous = scinp.random.uniform(low=50, high=100, size=(100, 4))

latent_dim = 2
num_layers = 2
upscaling_dimension = 4
batch_size = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

quantum_gen = QuantumGenerator(latent_dim, num_layers, upscaling_dimension, device)
quantum_gen = quantum_gen.to(device)


critic = Critic(data_dimension=upscaling_dimension)
critic = critic.to(device)

quantum_gen, critic = train_quantum_anogan(X_normal, quantum_gen, critic, device, n_iter=10, batch_size=64, n_critic=5)


generated_samples = [quantum_gen.forward().cpu().detach().numpy() for i in range(10)]
for i in range(len(generated_samples)):
    print("Sample {}:\t{}\n".format(i + 1, generated_samples[i]))
