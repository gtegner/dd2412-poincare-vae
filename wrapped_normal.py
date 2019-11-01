import numpy as np

import torch
from torch.distributions import Normal
import helpers
from helpers import norm_, lambda_x, exp_map


class WrappedNormal:
    def __init__(self, mu, sigma, c=1, name='wr'):
        super().__init__()
        self.name = name
        if len(mu.shape) == 2:
            self.batch_shape = mu.shape[0]
            self.dim = mu.shape[1]
        elif len(mu.shape) == 1:
            self.batch_shape = 1
            self.dim = mu.shape[0]

        self.mu = mu
        self.sigma = sigma
        self.c = c

        assert self.test(mu), "Mu must be in the ball"

        self.EPS = 1e-6

    def sample_n(self, n):
        norm = Normal(torch.zeros_like(self.mu), self.sigma)
        v = norm.rsample((n,))

        lambda_mu = lambda_x(self.mu, self.c, keepdim=True)

        z = exp_map(self.mu, v / lambda_mu, c=self.c)
        return z

    def rsample(self):
        mc_sample = 1
        return self.sample_n(mc_sample)

    @staticmethod
    def __safe_log__(x, eps):
        return torch.log(x + eps)

    def safe_log(self, x):
        return self.__safe_log__(x, self.EPS)

    @property
    def Z_w(self):
        return (np.sqrt(2 * np.pi) * self.sigma)**(-self.dim)

    @property
    def log_Zw(self):
        # Actually log(1/Z_w)
        log_z = (-self.dim) * \
            (np.log(np.sqrt(2 * np.pi)) + self.safe_log(self.sigma))
        return log_z.unsqueeze(0)

    def density(self, x):
        assert self.test(x), "x has got to be in the poincare ball"

        dist = helpers.dist_p(self.mu, x, self.c)
        dens = self.Z_w * torch.exp(-dist**2 / (2 * self.sigma**2)) * (
            dist * np.sqrt(self.c) / (torch.sinh(np.sqrt(self.c) * dist)))**(self.dim-1)

        dens[torch.isnan(dens)] = 0
        return dens

    def log_prob(self, x):

        dist = helpers.dist_p(self.mu, x, self.c, keepdim=True)
        sinh = torch.sinh(np.sqrt(self.c) * dist)
        part1 = self.log_Zw - (dist**2) / (2 * self.sigma**2)

        part2 = (self.dim-1) * (0.5 * np.log(self.c) +
                                self.safe_log(dist) - self.safe_log(sinh))

        log_dens = part1 + part2

        return log_dens

    def test(self, x):
        return test_in_ball(x, self.c)


def test_in_ball(x, c):
    in_ball = (norm_(x, 2) < 1.0 / np.sqrt(c)).all()
    return in_ball
