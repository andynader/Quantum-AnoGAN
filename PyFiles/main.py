from QuantumGenerator import *
from Critic import *
from ClassicalGenerator import *
from Training import *
from Utility import *

X_normal = generate_sinusoidal_data(n_points=1000)
latent_dim = 2
num_quantum_layers = 2
data_dimension = X_normal.shape[1]
batch_size = 64
n_iter = 20000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

# quantum_gen = QuantumGenerator(latent_dim, num_layers, upscaling_dimension, device)
# quantum_gen = quantum_gen.to(device)


critic_hidden_layer_sizes = [3, 3]
critic = Critic(data_dimension=data_dimension, hidden_layer_sizes=critic_hidden_layer_sizes)
critic = critic.to(device)

hidden_layer_sizes = [3, 3]
generator = ClassicalGenerator(latent_dim, hidden_layer_sizes, data_dimension, device)
generator = generator.to(device)
generator, critic = train_anogan(X_normal, generator, critic, device,
                                 n_iter=n_iter, batch_size=batch_size, n_critic=5, n_monitoring_samples=15)

# We print out the critic score on the fake data.


X_fake_small = torch.normal(mean=-100., std=1., size=(10, data_dimension)).to(device)

critic_score_fake_small = critic(X_fake_small).flatten().cpu().detach().numpy()
print(critic_score_fake_small)

X_fake_large = torch.normal(mean=100, std=5, size=(10, data_dimension)).to(device)
critic_score_fake_large = critic(X_fake_large).flatten().cpu().detach().numpy()

print(critic_score_fake_large)

plot_generator_3D(generator, n_points=2000)
