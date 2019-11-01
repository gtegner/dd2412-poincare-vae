import numpy as np
import torch
from helpers import dist_p, sample_from_hypersphere, exp_map, lambda_x
import helpers
from ars import ARS
from scipy import special
from utils import safe_log
import math

infty = torch.tensor(float('Inf'))
EPS = 1e-5


def get_signs_range(sigma, dim):
    num_range = torch.arange(dim).float()
    num_range = num_range.repeat(sigma.shape).T.unsqueeze(-1)
    signs = (-1)**num_range
    return signs, num_range


def __to_tensor__(d):
    if not isinstance(d, torch.Tensor):
        d = torch.tensor(d).float()
    return d


class RiemannianNormal:

    def __init__(self, mu, sigma, c, dim=None):
        self.batch_size = None
        if len(mu.shape) == 2:
            self.dim = mu.shape[1]
            self.batch_size = mu.shape[0]
        elif len(mu.shape) == 1:
            self.dim = mu.shape[0]

        self.c = __to_tensor__(c)
        self.sigma = sigma
        self.sigma = torch.clamp(sigma, min=.1)
        self.mu = mu
        if dim is not None:
            self.dim = dim

        self.EPS = 1e-6

    def sample_n(self, n):
        alpha = sample_from_hypersphere(n, self.dim, self.batch_size)
        r = self.sample_r(n)

        v = r * alpha

        lambda_mu = lambda_x(self.mu, self.c).unsqueeze(-1)
        lambda_mu = lambda_mu.view((1, *lambda_mu.shape))
        norm_v = v / lambda_mu
        norm_v = norm_v.squeeze(0)
        z = exp_map(self.mu, norm_v, c=self.c)
        return z

    def rsample(self):
        return self.sample_n(n=1)

    def sample_r(self, n):
        r = self.adaptive_rejection_sampling(n)
        return implicit_reparam(r, self.sigma, self.c, self.dim)

    # This is a quick fix and accounts for the case when x is of dim [num_samples, batch, dim]. Used for marginal likelihood calculation
    def log_prob_marginal(self, x):

        mu = self.mu.expand(x.shape)
        dist_sq = (dist_p(mu, x, c=self.c)**2)

        dist_sq = dist_sq.unsqueeze(-1)

        Z_r = self.Z_R(sigma=self.sigma, dim=self.dim, c=self.c)
        out = -dist_sq / (2*self.sigma**2) - safe_log(Z_r)
        return out

    def log_prob(self, x):

        mu = self.mu.expand(x.shape)
        dist_sq = (dist_p(mu, x, c=self.c)**2).T

        dist_sq = dist_sq.view(x.shape[0], len(dist_sq), 1)

        Z_r = self.Z_R(sigma=self.sigma, dim=self.dim, c=self.c)
        out = -dist_sq / (2*self.sigma**2) - safe_log(Z_r)
        return out

    def Z_R(self, sigma, dim, c):
        return self.Z_alpha(dim) * self.Z_r(sigma, dim, c)

    def rmu(self):
        return self.__rmu__(self.sigma, self.c, self.dim)

    def rstd(self):
        return torch.sqrt(self.var())

    def var(self):
        return self.__var__(self.sigma, self.c, self.dim)

    def cdf_r(self, r):
        return self.__cdf_r__(r, self.sigma, self.dim, self.c)

    def grad_log_prob(self, r):
        c = __to_tensor__(self.c)
        dim = self.dim
        res = - r / self.sigma.pow(2) + (dim - 1) * c.sqrt() * \
            torch.cosh(c.sqrt() * r) / torch.sinh(c.sqrt() * r)
        res[r < 0] = 0.0
        return res

    def adaptive_rejection_sampling(self, sample_shape=torch.Size()):

        with torch.no_grad():
            # Removed mu.clone()
            mu = self.rmu()
            std = self.rstd()
            if torch.isnan(std).any():
                print("std is nan")
                std[torch.isnan(std)] = self.sigma[torch.isnan(std)]
            if torch.isnan(mu).any():
                print("mu is nan")
                mu[torch.isnan(mu)] = (
                    (self.dim - 1) * self.sigma.pow(2) * np.sqrt(self.c))[torch.isnan(mu)]

            n_min, n_max, K = 0.1, 3, 20

            a = torch.linspace(n_max, n_min, int(K/2))
            b = torch.linspace(n_min, n_max, int(K/2))
            n = torch.cat((-a, b))
            xk = mu + n * torch.min(std, 0.95 * mu / n_max)
            ars = ARS(self.log_density_r,
                      self.grad_log_prob, xi=xk, ns=20, lb=0)
            samples = ars.sample(sample_shape)
        return samples

    @staticmethod
    def __var__(sigma, c, dim):
        moment1_sqr = RiemannianNormal.__rmu__(sigma, c, dim)**2

        dim = __to_tensor__(dim)
        c = __to_tensor__(c)

        zr = RiemannianNormal.Z_r(sigma, dim, c)
        constant = 1/(2 * np.sqrt(c))**(dim-1)

        signs, num_range = get_signs_range(sigma, dim)

        constant = constant.double()
        sigma = sigma.double()
        dim = dim.double()
        c = c.double()
        signs = signs.double()
        num_range = num_range.double()
        zr = zr.double()

        log_vec = RiemannianNormal.log_var_inner_sum(sigma, c, dim, num_range)
        log_sum = helpers.log_sum_exp_sign(log_vec, signs, dim=0)

        moment2 = 1/zr * constant * torch.exp(log_sum)

        out = moment2 - moment1_sqr.double()
        return out.float()

    @staticmethod
    def __rmu__(sigma, c, dim):

        zr = RiemannianNormal.Z_r(sigma, dim, c)

        constant = 1/(2 * np.sqrt(c))**(dim-1)

        signs, num_range = get_signs_range(sigma, dim)
        log_vec = RiemannianNormal.log_mu_inner_sum(sigma, c, dim, num_range)

        log_sum = helpers.log_sum_exp_sign(log_vec, signs, dim=0)

        return 1/zr * constant * torch.exp(log_sum)

    @staticmethod
    def log_mu_inner_sum(sigma, c, dim, k):
        dim = __to_tensor__(dim)

        a = torch.lgamma(dim) - torch.lgamma(dim-k)-torch.lgamma(k+1)
        b = (dim-1-2*k).pow(2)*c*sigma.pow(2) / 2

        # Integral
        mu = (dim-1-2*k)*torch.sqrt(c) * sigma.pow(2)

        int_r_a = 2 * sigma**2 * \
            torch.exp(-mu**2 / (2 * sigma**2))

        int_r_b = np.sqrt(np.pi / 2) * mu * sigma * \
            (1 + torch.erf(mu / (np.sqrt(2) * sigma)))

        log_int_r = torch.log(int_r_a + int_r_b)

        return a + b + log_int_r

    @staticmethod
    def log_var_inner_sum(sigma, c, d, k):
        d = __to_tensor__(d)

        a = torch.lgamma(d) - torch.lgamma(d-k)-torch.lgamma(k+1)
        b = (d-1-2*k).pow(2)*c*sigma.pow(2) / 2

        mu = (d-1-2*k) * np.sqrt(c) * sigma.pow(2)

        int_ = np.sqrt(2)/2 * sigma.pow(3) * (np.sqrt(np.pi) * (1 + torch.erf(mu/(np.sqrt(2)*sigma))) - 2 * torch.exp(-mu**2/(2 * sigma**2)) * (mu / (np.sqrt(2) * sigma))
                                              ) + 2 * sigma.pow(2) * mu * torch.exp(-mu**2 / (2 * sigma**2)) + np.sqrt(np.pi / 2) * sigma*mu.pow(2) * (1 + torch.erf(mu / (np.sqrt(2) * sigma)))

        log_int = torch.log(int_)

        return a + b + log_int

    @staticmethod
    def log_Z_r(sigma, d, c):

        if d == 2:
            return RiemannianNormal.log_Z_r_00(sigma, c)

        log_a = (np.log(np.pi) - np.log(2)) / 2 + \
            torch.log(sigma) - (d-1)*(np.log(2) + np.log(c) / 2)

        signs, num_range = get_signs_range(sigma, d)
        log_vec = RiemannianNormal.log_inner_sum(sigma, num_range, d, c)

        log_sum = helpers.log_sum_exp_sign(log_vec, signs, dim=0)

        return log_a + log_sum

    @staticmethod
    def Z_r(sigma, d, c):
        return torch.exp(RiemannianNormal.log_Z_r(sigma, d, c))

    @staticmethod
    def Z_r_00(sigma):
        return np.sqrt(np.pi / 2) * sigma * torch.exp(sigma**2 / 2) * torch.erf(sigma / np.sqrt(2))

    @staticmethod
    def log_Z_r_00(sigma, c):
        return 0.5 * np.log(np.pi / 2) + torch.log(sigma) - 0.5*torch.log(c) + c*sigma**2 / 2 + torch.log(torch.erf(sigma * torch.sqrt(c)/np.sqrt(2)))

    @staticmethod
    def log_inner_sum(sigma, k, d, c):
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(d).float()

        a = torch.lgamma(d) - torch.lgamma(d-k)-torch.lgamma(k+1)
        b = (d-1-2*k)**2*c*sigma**2 / 2
        c = safe_log(1 + torch.erf(((d-1-2*k)*np.sqrt(c)*sigma) / np.sqrt(2)))

        return a + b + c

    @staticmethod
    def Z_alpha(d):
        return 2 * (np.pi**(d / 2)) / (special.gamma(d / 2))

    @staticmethod
    def density_r(r, sigma, dim, c):
        return torch.exp(RiemannianNormal.__log_density_r__(r, sigma, dim, c))

    def log_density_r(self, r):
        return self.__log_density_r__(r, self.sigma, self.dim, self.c)

    @staticmethod
    def __log_density_r__(r, sigma, d, c):
        log_zr = RiemannianNormal.log_Z_r(sigma, d, c)
        log_exp = -r**2/(2*sigma**2)
        sinh_term = (d-1) * \
            torch.log(torch.sinh((np.sqrt(c) * r)) / (np.sqrt(c)))
        res = log_exp + sinh_term - log_zr
        res[res != res] = -infty
        return res

    @staticmethod
    def __cdf_r__(r, sigma, dim, c):
        r = r.double()
        sigma = sigma.double()
        c = c.double()

        if dim == 2:
            return 1 / torch.erf(c.sqrt() * sigma / np.sqrt(2)) * .5 * \
                (2 * torch.erf(c.sqrt() * sigma / np.sqrt(2)) + torch.erf((r - c.sqrt() * sigma**2) / math.sqrt(2) / sigma) -
                 torch.erf((torch.sqrt(c) * sigma**2) + r) / np.sqrt(2) / sigma)

        dim = __to_tensor__(dim)
        c = __to_tensor__(c)

        zr = RiemannianNormal.Z_r(sigma, dim, c)

        constant = np.sqrt(np.pi / 2) * 1 / (2 * np.sqrt(c))**(dim-1)
        r = r.double()
        sigma = sigma.double()
        c = c.double()
        signs, num_range = get_signs_range(sigma, dim)
        signs = signs.double()
        num_range = num_range.double()
        zr = zr.double()
        constant = constant.double()

        sum_vec = RiemannianNormal.cdf_log_inner_sum(
            r, sigma, dim, num_range, c)
        log_sum = helpers.log_sum_exp_sign(sum_vec, signs, dim=0)

        output = 1/zr * constant * sigma * torch.exp(log_sum)
        mask_zero = (r == 0).int()
        output = output * (1 - mask_zero).double()
        output[output != output] = 0
        assert not torch.isnan(output).any()
        return output.double()


    @staticmethod
    def cdf_log_inner_sum(r, sigma, d, k, c):
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(d).float()

        a = torch.lgamma(d) - torch.lgamma(d-k) - torch.lgamma(k+1)
        b = (d - 1 - 2 * k)**2 / 2 * c * sigma.pow(2)

        erf_term = torch.log(
            torch.erf((r - (d - 1 - 2 * k) * np.sqrt(c)
                       * sigma.pow(2)) / sigma / np.sqrt(2))
            + torch.erf((d - 1 - 2 * k) *
                        np.sqrt(c) * sigma / np.sqrt(2))
        )

        return a + b + erf_term

    @staticmethod
    def __grad_log_prob__(r, sigma, d, c):

        grad = - r / sigma**2 + \
            (d-1) * np.sqrt(c) * torch.cosh(np.sqrt(c) * r) / \
            (torch.sinh(np.sqrt(c) * r))

        grad[r < 0] = 0.0
        return grad


class ImplicitReparametrization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, sigma, c, dim, F):

        ctx.dim = dim
        density = RiemannianNormal.density_r(r, sigma, dim, c)
        ctx.save_for_backward(r, sigma, density, c)
        ctx.F = F

        return r

    @staticmethod
    def backward(ctx, grad_output):
        r, sigma, density, c = ctx.saved_tensors
        dim = ctx.dim
        F = ctx.F
        assert not torch.isnan(r).any()
        assert not torch.isnan(sigma).any()
        assert not torch.isnan(density).any()

        jacobian = torch.zeros(*r.shape).double()

        with torch.enable_grad():
            xv = r.detach()
            dv = density.detach().double()
            c = c.detach()
            sigma = sigma.detach().requires_grad_()

            z = F(xv, sigma, dim, c)
            for i in range(r.shape[0]):
                masked = torch.zeros_like(grad_output)
                masked[i, :, :] = 1.0
                grad_masked = (grad_output * masked).double()
                z.backward(grad_masked, retain_graph=True)
                jacobian[i] = sigma.grad.data
                sigma.grad.data.zero_()

        grad = -jacobian * dv.pow(-1)
        return None, grad.float(), None, None, None


def implicit_reparam(r, sigma, c, dim):
    F = RiemannianNormal.__cdf_r__
    return ImplicitReparametrization.apply(r, sigma, c, dim, F)

