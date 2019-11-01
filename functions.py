import torch
from scipy import special

MIN_NORM = 1e-15


class InvTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        z = z.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(z)
        return (torch.log_(1+z).sub_(torch.log_(1-z))).mul_(0.5)

    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[0]
        return (grad_output * 1.0/(1 - z**2))


class InvCosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        z = z.clamp_min_(1.0 + MIN_NORM)
        ctx.save_for_backward(z)
        return torch.log(z + torch.sqrt(1+z)*torch.sqrt(z-1))

    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[0]
        return grad_output * 1.0 / (torch.sqrt(z-1) * torch.sqrt(z+1))


class InvSinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        return torch.log((z + torch.sqrt(1 + torch.pow(z, 2))).clamp_min_(MIN_NORM))

    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[0]
        return grad_output * 1.0 / (1 + torch.pow(z, 2)).sqrt_()


def inv_tanh(z):
    return InvTanh.apply(z)


def inv_cosh(z):
    return InvCosh.apply(z)


def inv_sinh(z):
    return InvSinh.apply(z)
