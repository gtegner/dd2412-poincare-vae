import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import inv_cosh, inv_sinh, inv_tanh

MIN_NORM = 1e-10


def batch_dot_product(x, y):
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
    return _mobius_add_(x, y, c)


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

    dot = batch_dot_product(x, y)  # N x M x 1

    if x.shape == y.shape:
        first_num = (1 + 2*c*dot + c*norm_y_sq)*x
        second_num = ((1-c * norm_x_sq) * y)
    else:
        first_num = (1 + 2*c*dot + c*norm_y_sq)*x
        second_num = ((1-c * norm_x_sq) * y.T).T
        a = dot
        b = x

    num = first_num + second_num
    den = 1 + 2*c*dot + (c**2)*norm_x_sq*norm_y_sq

    return num / den.clamp_min(MIN_NORM)


def lambda_x(x, c):
    norm_x = norm_(x, p=2)**2
    return 1.0 / (1 - c * norm_x).clamp_min(MIN_NORM)


def dist_p_cosh(x, y, c):
    sqrt_c = np.sqrt(c)
    norm_x = norm_(x, 2)**2
    norm_y = norm_(y, 2)**2

    numerator = 2*c*norm_(x-y, 2)**2
    denominator = (1 - c * norm_x)*(1 - c * norm_y)
    # print("denominator", denominator)

    return 1.0 / sqrt_c * inv_cosh(1.0 + numerator / denominator.clamp_min(MIN_NORM))

# From Hyperbolic Neural Networks


def dist_p(x, y, c):
    sqrt_c = np.sqrt(c)
    dist_c = inv_tanh(sqrt_c * norm_(mobius_add(-x, y, c)))
    return dist_c * 2 / sqrt_c


def exp_map(x, v, c):
    lx = lambda_x(x, c)
    norm_v = norm_(v).clamp_min(MIN_NORM)
    sqrt_c = np.sqrt(c)

    t = torch.tanh(sqrt_c * 0.5 * lx * norm_v)
    v_adj = 1/(sqrt_c * norm_v) * v

    return mobius_add(x, t * v_adj, c=c)


def exp_0_map(v, c):
    norm_v = norm_(v, p=2).clamp_min(MIN_NORM)
    return torch.tanh(np.sqrt(c) * norm_v) * v / (np.sqrt(c) * norm_v)


def log_map(x, y, c):
    """Exponential map exp^{c}_x(v)}

    Arguments:
        x torch.Tensor
        y torch.Tensor

    Keyword Arguments:
        c Poincare ball parameter 

    Returns:
        [type] -- [description]
    """
    lx = lambda_x(x, c)
    sqrt_c = np.sqrt(c)
    x_add_y = mobius_add(-x, y, c)
    norm_x_add_y = norm_(x_add_y).clamp_min(MIN_NORM)

    return 2.0 * 1/(sqrt_c * lx) * inv_tanh(sqrt_c * norm_x_add_y) * x_add_y / norm_x_add_y


def hyperplane_dist(x, a, b, c):
    sqrt_c = np.sqrt(c)
    xb = mobius_add(-b, x, c)

    dot = batch_dot_product(xb, a)
    numerator = 2 * sqrt_c * torch.abs(dot)
    norm_xb = norm_(xb, 2)**2

    norm_a = norm_(a).clamp_min(MIN_NORM)
    denominator = (1 - c * norm_xb) * norm_a

    return 1.0 / sqrt_c * inv_sinh(numerator / denominator.clamp_min(MIN_NORM))
