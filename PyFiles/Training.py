from QuantumGenerator import *
from Critic import *
from Utility import *


# All the sampled arrays returned are torch tensors.
def sample_arrays(X, generator, batch_size, data_dimension, device):
    # We sample n=batch_size points from the data
    row_indices = np.random.choice(len(X), size=batch_size)
    X_minibatch = X[row_indices]
    # We set requires grad to True since we need the gradient of the critic
    # output with respect to the input when calculating the critic loss.
    X_minibatch = torch.Tensor(X_minibatch).to(device)
    X_minibatch.requires_grad = True

    # We sample n=batch_size points from the generator
    gen_minibatch = torch.Tensor(0, data_dimension).to(device)
    for k in range(batch_size):
        sample = generator().unsqueeze(0)
        gen_minibatch = torch.cat((gen_minibatch, sample))

    # We sample n=batch_size noise points
    epsilons = torch.Tensor(batch_size).uniform_(0, 1).to(device)
    return X_minibatch, gen_minibatch, epsilons


def get_X_hat(X_minibatch, gen_minibatch, epsilons):
    eps_new_axis = epsilons[:, np.newaxis]
    X_hat = X_minibatch + eps_new_axis * (gen_minibatch - X_minibatch)
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


def critic_loss(X_minibatch, X_hat_minibatch, gen_minibatch, critic, device, l=10):
    critic_grad_x_hat = critic_grad_wrt_inputs(X_hat_minibatch, critic, device)
    L_c = critic(gen_minibatch) - critic(X_minibatch)
    L_c = L_c.flatten()
    grad_norm = torch.linalg.norm(critic_grad_x_hat, ord=2, dim=1)
    L_c += l * (grad_norm - 1) ** 2
    return L_c.mean()


def generator_loss(generator, critic, batch_size, device):
    data_dimension = generator.data_dimension
    gen_minibatch = torch.Tensor(0, data_dimension).to(device)
    for k in range(batch_size):
        sample = generator().unsqueeze(0)
        gen_minibatch = torch.cat((gen_minibatch, sample))

    L_g = -critic(gen_minibatch).flatten()
    return L_g.mean()


def train_anogan(X, generator, critic: Critic, device, n_iter=1, batch_size=64,
                 n_critic=5, n_monitoring_samples=3):
    num_features = X.shape[1]
    data_dimension = generator.data_dimension

    assert data_dimension == num_features
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0, 0.9))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0001, betas=(0, 0.9))
    for i in range(n_iter):
        if i % 100 == 0:
            print("Iteration ", i + 1)
        for j in range(n_critic):
            X_minibatch, gen_minibatch, epsilons = sample_arrays(X, generator, batch_size,
                                                                 data_dimension, device)

            X_hat_minibatch = get_X_hat(X_minibatch, gen_minibatch, epsilons)

            critic_optimizer.zero_grad()
            crit_loss = critic_loss(X_minibatch, X_hat_minibatch, gen_minibatch, critic, device)
            crit_loss.backward()
            critic_optimizer.step()

        generator_optimizer.zero_grad()
        gen_loss = generator_loss(generator, critic, batch_size, device)
        gen_loss.backward()
        generator_optimizer.step()
        # Uncomment or comment the next few lines for some logging
        # to help with debugging.
        if i % 100 == 0:
            # check_both_models_strength(X, critic, generator, device)
            # monitor_generated_samples(generator, n_monitoring_samples)
            # print_parameters(generator)
            plot_generator_3D(generator)
    return generator, critic
