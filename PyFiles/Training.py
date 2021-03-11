from Generator import *
from Critic import *


# All the sampled arrays returned are torch tensors.
def sample_arrays(X, generator, batch_size, upscaling_dimension, device):
    # We sample n=batch_size points from the data
    row_indices = np.random.choice(len(X), size=batch_size)
    X_minibatch = X[row_indices]
    X_minibatch = torch.Tensor(X_minibatch).to(device)

    # We sample n=batch_size points from the generator
    z_minibatch = torch.Tensor(0, upscaling_dimension).to(device)
    for k in range(batch_size):
        sample = generator().unsqueeze(0)
        z_minibatch = torch.cat((z_minibatch, sample))

    # We sample n=batch_size noise points
    epsilons = torch.Tensor(batch_size).uniform_(0, 1).to(device)
    return X_minibatch, z_minibatch, epsilons


def get_X_hat(X_minibatch, z_minibatch, epsilons):
    eps_new_axis = epsilons[:, np.newaxis]
    X_hat = eps_new_axis * (X_minibatch - z_minibatch)
    return X_hat


def critic_loss(X_minibatch, X_hat_minibatch, z_minibatch, critic, l=10):
    #TODO: Define the critic loss given in the paper
    pass


def train_quantum_anogan(X, generator: QuantumGenerator, critic: Critic, device, n_iter=1, batch_size=64,
                         n_critic=5):
    num_features = X.shape[1]
    upscaling_dimension = generator.upscaling_dimension

    assert upscaling_dimension == num_features

    for i in range(n_iter):
        for j in range(n_critic):
            X_minibatch, z_minibatch, epsilons = sample_arrays(X, generator, batch_size,
                                                               upscaling_dimension, device)

            X_hat = get_X_hat(X_minibatch, z_minibatch, epsilons)

            #TODO: Continue the rest of the algorithm here.
