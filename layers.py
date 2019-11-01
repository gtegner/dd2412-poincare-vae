import torch
import torch.nn as nn
import torch.nn.functional as F
from wrapped_normal import WrappedNormal, test_in_ball

from helpers import log_map, lambda_x, norm_, hyperplane_dist, exp_0_map
from geoopt.manifolds.poincare.math import inner
from torch.distributions import Normal, RelaxedBernoulli, MultivariateNormal

from riemannian_normal import RiemannianNormal


class EfficientHyperbolicLinear(nn.Module):
    def __init__(self, input_dim, output_dim, c):
        super().__init__()
        self.c = c

        self.a = nn.Parameter(torch.Tensor(
            output_dim, 1, input_dim))  # In tangent space Rn
        self.b = nn.Parameter(torch.Tensor(
            output_dim, 1, input_dim))  # In tangent space Rn

        self.__reset_weights__()

    def forward(self, x):
        """
        We have parameters
        x \in B_c^{d}
        a' \in T_0(B) = R^n
        b' \in T_0(B) = R^n

        We want
        a \in T_b(B)
        b \in B_c^{d}
        log_b(x) in T_b(D)

        Parallel transport a' -> a by
        b = exp_0(b')
        a = log_b(b + exp_0(a')) = (lambda_0 / lambda_b) a'
        Find log_b(x)

        Calculate
        <log_b(x),a>_(b) = (lambda_b)**2 <log_b(x), a>
        |a|_p = lambda_b|a|


        Arguments:
            x {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        exp_b = exp_0_map(self.b, self.c)
        log_b = log_map(exp_b, x, self.c)
        log_b = log_b.transpose(0, 1)
        lam_x_b = lambda_x(exp_b, self.c).unsqueeze(-1)
        parallel_a = 2 / lam_x_b * self.a

        exp_b = exp_b.squeeze(1)
        exp_b = exp_b.expand_as(log_b)
        parallel_a = parallel_a.squeeze(1)
        parallel_a = parallel_a.expand_as(log_b)

        sign_dot = torch.sign(
            inner(exp_b, log_b, parallel_a, c=self.c, keepdim=True))

        norm_a = lambda_x(self.b, self.c) * \
                 norm_(self.a).squeeze(-1)
        hp_dist = hyperplane_dist(
            x, self.a, self.b, self.c).unsqueeze(-1).transpose(0, 1)
        norm_a = norm_a.expand_as(sign_dot)
        out = sign_dot * norm_a * hp_dist
        return out.squeeze(-1)

    def __reset_weights__(self):
        self.a.data.uniform_(-0.01, 0.01)
        self.b.data.uniform_(-0.01, 0.01)


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, c, h=32):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, h), nn.ReLU())

        self.out = nn.Linear(h, latent_dim + 1)
        self.mu = nn.Linear(h, latent_dim)
        self.sigma = nn.Linear(h, 1)
        self.c = c

        self.softplus = nn.Softplus()
        self.latent_dim = latent_dim

    def forward(self, x):
        h = self.layers(x)
        if torch.isnan(h).any():
            print("h is non")
            raise ValueError

        mu = self.mu(h)
        sigma = self.sigma(h)
        if torch.isnan(mu).any():
            print("MU is non")
            raise ValueError

        mu_frechet = exp_0_map(mu, self.c)
        sigma_out = F.softplus(.5 * sigma)

        is_ball = test_in_ball(mu_frechet, self.c)
        if not is_ball:
            print("Frechet mu not in ball")
            raise ValueError

        return mu_frechet, sigma_out


class HyperbolicVAE(nn.Module):
    def __init__(self, input_dim,
                 c,
                 latent_dim=2,
                 dist='wrapped',
                 h=32,
                 gyroplane_dim=32,
                 prior_sigma=1.0,
                 loss='mse'):

        super().__init__()

        self.c = c
        self.dist = dist
        self.encoder = Encoder(input_dim, latent_dim, c, h)
        self.loss = loss

        self.decoder = nn.Sequential(EfficientHyperbolicLinear(
            latent_dim, gyroplane_dim, c), nn.ReLU(), nn.Linear(gyroplane_dim, input_dim))

        if loss in ['mse']:
            self.likelihood_dist = Normal
        elif loss in ['bce']:
            self.likelihood_dist = RelaxedBernoulli

        self.prior_dist = None
        self.prior_sigma = prior_sigma
        if dist in ['wrapped']:
            self.prior_dist = WrappedNormal(
                torch.zeros(latent_dim), torch.Tensor([prior_sigma]), name='prior', c=self.c)
        else:
            mu = torch.zeros((1, latent_dim)).requires_grad_(False)
            sigma = torch.Tensor([[prior_sigma]]).requires_grad_(False)
            self.prior_dist = RiemannianNormal(
                mu=mu, sigma=sigma, c=c)
            #self.prior_dist = None
            # raise NotImplementedError

    def forward(self, x):
        mu, sigma = self.encoder(x)

        in_ball = test_in_ball(mu, self.c)
        if not in_ball:
            print("Mu is not in ball")
            print(mu)
            raise ValueError

        if self.dist in ['wrapped']:
            normal = WrappedNormal(mu, sigma, name='posterior', c=self.c)
        else:
            normal = RiemannianNormal(mu, sigma, c=self.c)

        z = normal.rsample()  # batch x dim

        if self.dist in ['riemannian'] and len(z.shape) == 2:
            z = z.unsqueeze(0)

        log_prob = normal.log_prob(z)

        prior_log_prob = self.prior_dist.log_prob(z)

        if self.dist in ['riemannian']:
            z = z.squeeze(0)

        decoded = self.decoder(z)
        return decoded, mu, sigma, log_prob, prior_log_prob

    def neg_kl_div(self, log_prob, prior_log_prob):
        return log_prob.sum(-1) - prior_log_prob.sum(-1)

    def loss_fn(self, x):
        decoded, mu, sigma, log_prob, prior_log_prob = self.forward(x)

        lpx_z = None

        if self.loss in ['mse']:
            decoded = decoded.unsqueeze(0)
            px_z = self.likelihood_dist(*[decoded, torch.ones_like(decoded)])
            flat_rest = torch.Size([*px_z.batch_shape[:2], -1])

            lpx_z = px_z.log_prob(
                x.expand(px_z.batch_shape)).view(flat_rest).sum(-1)
        elif self.loss in ['bce']:

            lpx_z = - \
                F.binary_cross_entropy_with_logits(
                    decoded, x, reduction='none')

        neg_kl = self.neg_kl_div(log_prob, prior_log_prob)

        likelihood = -lpx_z.mean(0).sum()
        return likelihood, neg_kl.mean(0).sum()

    def generate_images(self, num_samples):
        z = self.prior_dist.sample_n(num_samples)
        generated = self.decoder(z)
        return generated, z


class VanillaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, h=200, prior_sigma=1):
        super().__init__()

        hidden_dim = h
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.sigma = nn.Linear(hidden_dim, 1)

        self.fc4 = nn.Linear(latent_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)

        self.latent_dim = latent_dim
        self.softplus = nn.Softplus()
        self.prior_sigma = prior_sigma

        self.likelihood_dist = Normal

        self.prior_dist = MultivariateNormal(torch.Tensor(
            torch.zeros((latent_dim))), torch.eye(latent_dim) * prior_sigma)

        self.dist = 'normal'

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))

        mu = self.mu(h1)
        sigma = self.sigma(h1)

        sigma = self.softplus(sigma)

        return mu, sigma

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(mu)*self.prior_sigma

        return mu + eps*sigma

    def decoder(self, z):
        h3 = F.relu(self.fc4(z))
        return self.fc5(h3)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        return self.decoder(z), mu, sigma

    def loss_fn(self, x):
        decoded, mu, sigma = self.forward(x)
        dist = self.likelihood_dist(decoded, torch.ones_like(decoded))

        likelihood = dist.log_prob(x)
        neg_kl_div = -0.5 * \
            torch.sum(1 + 2 * torch.log(sigma) - mu**2 - sigma**2)

        return -likelihood.sum(), neg_kl_div




