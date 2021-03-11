from Generator import *
from Critic import *


# All the sampled arrays returned are torch tensors.
def sample_arrays(X, generator, batch_size, upscaling_dimension, device):
    # We sample n=batch_size points from the data
    row_indices = np.random.choice(len(X), size=batch_size)
    X_minibatch = X[row_indices]
    # We set requires grad to True since we need the gradient of the critic
    # output with respect to the input when calculating the critic loss.
    X_minibatch = torch.Tensor(X_minibatch).to(device)
    X_minibatch.requires_grad = True

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


def critic_grad_wrt_inputs(X_hat_minibatch, critic, device):
    num_features = X_hat_minibatch.shape[1]
    gradients = torch.Tensor(0, num_features).to(device)
    for i in range(len(X_hat_minibatch)):
        X_i = X_hat_minibatch[i]
        y_i = critic(X_i)
        grad_i = torch.autograd.grad(outputs=y_i, inputs=X_i)[0].unsqueeze(0)
        grad_i = grad_i.to(device)
        gradients = torch.cat((gradients, grad_i))
    return gradients


def critic_loss(X_minibatch, X_hat_minibatch, z_minibatch, critic, device, l=10):
    N = len(X_minibatch)
    critic_grad_x_hat = critic_grad_wrt_inputs(X_hat_minibatch, critic, device)
    L_c = critic(z_minibatch) - critic(X_minibatch)
    L_c = L_c.flatten()
    grad_norm = torch.linalg.norm(critic_grad_x_hat, ord=2, dim=1)
    L_c += l * (grad_norm - 1) ** 2
    return torch.sum(L_c) / N


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

            # TODO: Continue the rest of the algorithm here.
