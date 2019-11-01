import geoopt.manifolds.poincare.math as pmath
import torch
import numpy as np
from torch.distributions import Normal, MultivariateNormal

MIN_NORM = 1e-10
EPS = 1e-6


def batch_dot_product(x, y):
    print("batch dot product called")
    """Dot product between [N x d], [MxD] size vectors

    Arguments:
        x Batch x dim vector
        y Batch x dim vector
        return: torch.Tensor of size [N x M x 1]
    """

    if len(x.shape) == 1 and len(y.shape) == 1:
        return torch.sum(x * y)
    # if len(x.shape) == 2 and len(y.shape) == 2 and
    if x.shape == y.shape:
        return torch.sum(x * y, -1, keepdim=True)

    elif len(x.shape) == 1:
        # print("xhape 1")
        # print("xshape", x.shape)
        # print("yshape", y.shape)
        return torch.sum(x * y, -1, keepdim=True)

    out = (x @ y.T).unsqueeze(-1)
    # print("OUT")
    # print("xshape", x.shape)
    # print("yshape", y.shape)

    return out

    # return torch.bmm(x.view(n, 1, d), y.view(n, d, 1)).squeeze(-1)


def norm_(x, p=2, keepdim=True):
    return torch.norm(x, p, dim=-1, keepdim=keepdim)


def mobius_add(x, y, c):
    return pmath.mobius_add(x, y, c=c)
    # return _mobius_add_(x, y, c)


def _mobius_add_(x, y, c):
    """[summary]

    Arguments:
        x N x D
        y M x D
        c float

    Returns:
        [type] -- [description]
    """
    norm_x_sq = norm_(x, p=2)**2
    norm_y_sq = norm_(y, p=2)**2

    # print("norm x")
    # print(norm_x_sq.shape)
    # print("norm y")
    # print(norm_y_sq.shape)

    dot = batch_dot_product(x, y)  # N x M x 1
    print("\n")
    print("x", x.shape)
    print("y", y.shape)

    if x.shape == y.shape:
        first_num = (1 + 2*c*dot + c*norm_y_sq)*x
        second_num = ((1-c * norm_x_sq) * y)
    else:
        # print("ELSE")
        # print(x.shape)
        # print(y.shape)
        first_num = (1 + 2*c*dot + c*norm_y_sq)*x
        second_num = ((1-c * norm_x_sq) * y.T).T
        a = dot
        b = x
        print("a", a.shape)
        print("b", b.shape)

    print("first num", first_num.shape)
    print("second num", second_num.shape)

    num = first_num + second_num
    # print("num shape", num.shape)
    den = 1 + 2*c*dot + (c**2)*norm_x_sq*norm_y_sq
    # print("den shape", den.shape)

    return num / den.clamp_min(MIN_NORM)


def lambda_x(x, c, keepdim=False):
    return pmath._lambda_x(x, c, keepdim=keepdim)
    # norm_x = norm_(x, p=2)**2
    # return 1.0 / (1 - c * norm_x).clamp_min(MIN_NORM)


def dist_p_cosh(x, y, c):
    sqrt_c = np.sqrt(c)
    norm_x = norm_(x, 2)**2
    norm_y = norm_(y, 2)**2
    # print("in dist")
    # print("norm_x", norm_x)
    # print("norm_y", norm_y)

    numerator = 2*c*norm_(x-y, 2)**2
    denominator = (1 - c * norm_x)*(1 - c * norm_y)
    # print("denominator", denominator)

    return 1.0 / sqrt_c * inv_cosh(1.0 + numerator / denominator.clamp_min(MIN_NORM))

# From Hyperbolic Neural Networks


def dist_p(x, y, c, keepdim=False, dim=-1):
    return pmath._dist(x, y, c, keepdim, dim)


def exp_map(x, v, c):
    v = v + EPS
    return pmath.expmap(x, v, c=c, dim=-1)

    lx = lambda_x(x, c)
    norm_v = norm_(v).clamp_min(MIN_NORM)
    sqrt_c = np.sqrt(c)

    t = torch.tanh(sqrt_c * 0.5 * lx * norm_v)
    v_adj = 1/(sqrt_c * norm_v) * v

    a = pmath.expmap(x, v, c=c, dim=-1)
    return project_hyp_vecs(a, c)
    # return mobius_add(x, t * v_adj, c=c)


def exp_0_map(v, c, dim=-1):
    # #norm_v = norm_(v, p=2).clamp_min(MIN_NORM)
    # return torch.tanh(np.sqrt(c) * norm_v) * v / (np.sqrt(c) * norm_v)
    a = pmath.expmap0(v, c=c, dim=dim)
    return project_hyp_vecs(a, c)


def project_hyp_vecs(x, c):
        # https://www.tensorflow.org/api_docs/python/tf/clip_by_norm
    # Projection op. Need to make sure hyperbolic embeddings are inside the unit ball.
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c)
    norm_x = norm_(x)
    clip_norm = (1. - 1e-5) / torch.sqrt(c)
    intermediate = x * clip_norm
    out = intermediate / torch.max(norm_x, clip_norm)
    return out


def log_map(x, y, c, dim=-1):
    """Exponential map exp^{c}_x(v)}

    Arguments:
        x torch.Tensor
        y torch.Tensor

    Keyword Arguments:
        c Poincare ball parameter

    Returns:
        [type] -- [description]
    """
    # lx = lambda_x(x, c)
    # sqrt_c = np.sqrt(c)
    # x_add_y = mobius_add(-x, y, c)
    # norm_x_add_y = norm_(x_add_y).clamp_min(MIN_NORM)

    # return 2.0 * 1/(sqrt_c * lx) * inv_tanh(sqrt_c * norm_x_add_y) * x_add_y / norm_x_add_y

    return pmath.logmap(x, y, c=c, dim=dim)


def hyperplane_dist(x, a, b, c, keepdim=False):
    return pmath.dist2plane(x, b, a, keepdim=keepdim, c=c)
    # sqrt_c = np.sqrt(c)
    # xb = mobius_add(-b, x, c)

    # dot = batch_dot_product(xb, a)
    # numerator = 2 * sqrt_c * torch.abs(dot)
    # norm_xb = norm_(xb, 2)**2

    # norm_a = norm_(a).clampmin(MIN_NORM)
    # denominator = (1 - c * norm_xb) * norm_a

    # return 1.0 / sqrt_c * inv_sinh(numerator / denominator.clamp_min(MIN_NORM))


def sample_from_hypersphere(n, dim, batch_size=None):
    shape = torch.Size([n, batch_size, dim])
    output = torch.distributions.utils._standard_normal(
        shape, dtype=torch.float, device='cpu')
    return output / output.norm(dim=-1, keepdim=True)


def log_sum_exp_sign(log_vec, signs, dim=0):
    m, _ = torch.max(log_vec, dim=dim, keepdim=True)
    v0 = log_vec - m
    m = m.squeeze(dim)
    return m + torch.log(torch.sum(signs * torch.exp(v0), dim=dim, keepdim=False))


def log_sum_exp(log_vec, dim=0):
    m, _ = torch.max(log_vec, dim=dim, keepdim=True)
    v0 = log_vec - m
    m = m.squeeze(dim)

    return m + torch.log(torch.sum(torch.exp(v0), dim=dim, keepdim=False)) - np.log(log_vec.size(dim))


def marginal_likelihood(model, x, num_samples, mode='importance'):
    print("Calculating Marginal Likelihood")
    if mode in ['importance']:
        return importance_sampling_marginal(model, x, num_samples)
    elif mode in ['monte_carlo']:
        return monte_carlo_marginal(model, x, num_samples)


def importance_sampling_marginal(model, x, num_samples):
    from riemannian_normal import RiemannianNormal
    from wrapped_normal import WrappedNormal
    batch_size = x.shape[0]

    mu, sigma = model.encoder(x)

    if model.dist in ['wrapped']:
        normal = WrappedNormal(mu, sigma, name='posterior', c=model.c)
    elif model.dist in ['riemannian']:
        normal = RiemannianNormal(mu, sigma, c=model.c)
    elif model.dist in ['normal']:
        normal = Normal(mu, sigma)

    posterior_samples = normal.sample_n(num_samples)

    if model.dist not in ['riemannian']:
        log_posterior = normal.log_prob(posterior_samples)
    else:
        log_posterior = normal.log_prob_marginal(posterior_samples)

    if model.dist in ['normal']:
        log_posterior = log_posterior.sum(-1, keepdim=False)

    posterior_samples = posterior_samples.view(batch_size * num_samples, -1)

    # Get prior log likelihood
    if model.dist not in ['riemannian']:
        prior_lik = model.prior_dist.log_prob(posterior_samples)
    else:
        prior_lik = model.prior_dist.log_prob_marginal(posterior_samples)

    prior_lik = prior_lik.view(num_samples, batch_size, 1).sum(-1)

    if model.dist in ['riemannian']:
        posterior_samples = posterior_samples.squeeze(1)

    decoded = model.decoder(posterior_samples)
    decoded = decoded.view(num_samples, batch_size, -1)

    # Get log likelihood
    px_z = model.likelihood_dist(*[decoded, torch.ones_like(decoded)])
    flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
    x_expanded = x.expand(num_samples, *x.size())

    log_lik = px_z.log_prob(x_expanded).view(flat_rest).sum(-1)

    if model.dist in ['riemannian', 'wrapped']:
        log_posterior = log_posterior.sum(-1)

    obj = log_lik + prior_lik - log_posterior
    sum_exp = log_sum_exp(obj, 0)
    return -sum_exp.sum()


def monte_carlo_marginal(model, x, num_samples):
    print("Calculating Marginal Likelihood by monte-carlo sampling")

    batch_size = x.shape[0]

    # Sample z
    z = model.prior_dist.sample_n(num_samples * batch_size)

    # Get prior log likelihood
    prior_lik = model.prior_dist.log_prob(z)
    prior_lik = prior_lik.view(num_samples, batch_size, 1).sum(-1)

    if model.dist in ['riemannian']:
        z = z.squeeze(1)

    decoded = model.decoder(z)
    decoded = decoded.view(num_samples, batch_size, -1)

    # Get log likelihood
    px_z = model.likelihood_dist(*[decoded, torch.ones_like(decoded)])
    flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
    x_expanded = x.expand(num_samples, *x.size())

    log_lik = px_z.log_prob(x_expanded).view(flat_rest).sum(-1)

    # log p(x|z) + log p(z)
    obj = log_lik + prior_lik

    sum_exp = log_sum_exp(obj, 0)

    return -sum_exp.mean()
