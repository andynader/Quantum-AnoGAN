from QuantumGenerator import *
from Critic import *
import unittest


class TestQuantumAnoGAN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cls.latent_dim = 2
        cls.num_quantum_layers = 2
        cls.upscaling_dimension = 4
        cls.device = device
        quantum_gen = QuantumGenerator(cls.latent_dim, cls.num_quantum_layers,
                                       cls.upscaling_dimension, cls.device)
        quantum_gen = quantum_gen.to(device)
        cls.quantum_gen = quantum_gen

        critic = Critic(data_dimension=cls.upscaling_dimension)
        cls.critic = critic.to(device)

        # The generator and the discriminator/critic can have completely different
        # optimizers in a GAN. In fact, they can be completely different
        # learning algorithms, and the only requirement is that the
        # the output dimension of the first matches the input dimension of the
        # second. Thus, we define two different ADAM optimizers for each network.

        cls.generator_optimizer = torch.optim.Adam(quantum_gen.parameters())
        cls.critic_optimizer = torch.optim.Adam(critic.parameters())

    # test_gen_output() makes sure that the output
    # of the quantum generator is a one dimensional
    # tensor with length equal to the data dimension.
    def test_gen_output(self):
        sample = self.quantum_gen()
        assert len(sample) == self.upscaling_dimension

    # test_critic_output() makes sure that when the critic
    # is fed a generated sample, it will output a a tensor
    # of length one, i.e a scalar.
    def test_critic_output(self):
        sample = self.quantum_gen()
        critic_output = self.critic(sample)
        assert len(critic_output) == 1

    # test_differentiable_params() makes sure that the optimizer is
    # seeing the quantum weights (along with all other generator weights)
    # as differentiable parameters, in order to avoid situations where
    # some set of parameters of the hybrid quantum classical algorithm
    # is not being optimized without us being aware of this.
    def test_differentiable_params(self):
        # For the generator, we have three sets of parameters:
        # 1) The quantum weights
        # 2) The upscaling layer weights.
        # 3) The upscaling layer bias
        # and thus, the length of the quantum generator's
        # parameters should be three.
        assert len(list(self.quantum_gen.parameters())) == 3

    # test_optimizers() makes sure that there is no problem with stepping
    # using the ADAM optimizers for the hybrid quantum classical generator.
    def test_optimizers(self):
        old_parameters = self.quantum_gen.parameters()
        y_true = torch.Tensor(4).uniform_(-5, 5).to(self.device)
        gen_output = self.quantum_gen()
        self.generator_optimizer.zero_grad()
        # We use toy losses.
        gen_loss = nn.functional.mse_loss(input=gen_output, target=y_true)
        gen_loss.backward()
        self.generator_optimizer.step()
        new_parameters = self.quantum_gen.parameters()
        for p in zip(old_parameters, new_parameters):
            # Check if same size, different values.
            assert p[0].shape == p[1].shape and not torch.equal(p[0], p[1])
