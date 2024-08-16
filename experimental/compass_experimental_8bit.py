import torch
from torch.optim import Optimizer
import torch.nn.functional as F
from einops import rearrange

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise


# @torch.compile
def quantize(tensor, group_size=8, eps=1e-8, factor=3.2):
    shape = tensor.shape
    numel = tensor.numel()

    # just in case it's not divisible by group size
    padding = numel % group_size

    if padding != 0:
        tensor = rearrange(
            F.pad(tensor.flatten(), (0, padding), "constant", 0), "(r g) -> r g", g=2
        )
    else:
        tensor = rearrange(tensor.flatten(), "(r g) -> r g", g=group_size)
    scale = tensor.abs().max(dim=-1).values.unsqueeze(dim=-1)
    tensor /= scale + eps
    sign = tensor.sign()

    tensor = (
        ((torch.pow(tensor.abs(), 1 / factor) * sign + 1) * 127.5)
        .round()
        .to(dtype=torch.uint8)
    )
    if padding != 0:
        tensor = tensor.flatten()[:-padding]
    tensor = tensor.view(shape)
    return tensor, (scale, group_size, eps, factor, padding)


# @torch.compile
def dequantize(tensor, details, dtype=torch.float32):
    scale, group_size, eps, factor, padding = details
    shape = tensor.shape

    if padding != 0:
        tensor = rearrange(
            F.pad(tensor.flatten(), (0, padding), "constant", 0), "(r g) -> r g", g=2
        )
    else:
        tensor = rearrange(tensor.flatten(), "(r g) -> r g", g=group_size)
    tensor = tensor.to(dtype=dtype) / 127.5 - 1
    sign = tensor.sign()
    tensor = torch.pow(tensor.abs(), factor) * sign * scale
    if padding != 0:
        tensor = tensor.flatten()[:-padding]
    tensor = tensor.view(shape)

    return tensor


class CompassExperimental8Bit(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
        quantization_group_size (int):
            number of quant group (default: 8).
        quantization_factor (float):
            non linear quantization using x^f (default: 3.2)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        amp_fac=2,
        eps=1e-8,
        weight_decay=0,
        centralization=0,
        quantization_group_size=8,
        quantization_factor=3.2,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
            group_size=quantization_group_size,
            factor=quantization_factor,
        )
        super(CompassExperimental8Bit, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["ema"] = quantize(
                        torch.zeros_like(p.data),
                        group_size=group["group_size"],
                        factor=group["factor"],
                    )
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = quantize(
                        torch.zeros_like(p.data),
                        group_size=group["group_size"],
                        factor=group["factor"],
                    )

                beta1, beta2 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                state["step"] += 1

                # center the gradient vector
                if centralization != 0:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )

                # bias correction step size
                # soft warmup
                bias_correction = 1 - beta1 ** state["step"]
                bias_correction_sqrt = (1 - beta2 ** state["step"]) ** (1 / 2)
                step_size = lr / bias_correction

                # Decay the first and second moment running average coefficient
                ema = dequantize(*state["ema"]) + (1 - beta1) * grad
                # ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amplification_factor
                grad.add_(ema, alpha=amplification_factor)

                ema_squared = dequantize(*state["ema_squared"]) + (1 - beta2) * grad**2
                state["ema"] = quantize(
                    ema, group_size=group["group_size"], factor=group["factor"]
                )

                # ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])
                state["ema_squared"] = quantize(
                    ema_squared, group_size=group["group_size"], factor=group["factor"]
                )
                if weight_decay != 0:
                    # Perform stepweight decay
                    p.data.mul_(1 - step_size * weight_decay)

                # p = p - lr * grad / denom
                p.data.addcdiv_(grad, denom, value=-step_size)

        return loss


class CompassExperimental8BitBNB(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
        quantization_group_size (int):
            number of quant group (default: 64).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        amp_fac=2,
        eps=1e-8,
        weight_decay=0,
        centralization=0,
        quantization_group_size=64,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
            group_size=quantization_group_size,
        )
        super(CompassExperimental8BitBNB, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["ema"] = quantize_blockwise(
                        torch.zeros_like(p.data),
                        blocksize=group["group_size"],
                    )
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = quantize_blockwise(
                        torch.zeros_like(p.data),
                        blocksize=group["group_size"],
                    )

                beta1, beta2 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                state["step"] += 1

                # center the gradient vector
                if centralization != 0:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )

                # bias correction step size
                # soft warmup
                bias_correction = 1 - beta1 ** state["step"]
                bias_correction_sqrt = (1 - beta2 ** state["step"]) ** (1 / 2)
                step_size = lr / bias_correction

                # Decay the first and second moment running average coefficient
                ema = dequantize_blockwise(*state["ema"]) + (1 - beta1) * grad
                # ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amplification_factor
                grad.add_(ema, alpha=amplification_factor)

                ema_squared = (
                    dequantize_blockwise(*state["ema_squared"]) + (1 - beta2) * grad**2
                )
                state["ema"] = quantize_blockwise(
                    ema,
                    blocksize=group["group_size"],
                )

                # ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])
                state["ema_squared"] = quantize_blockwise(
                    ema_squared,
                    blocksize=group["group_size"],
                )
                if weight_decay != 0:
                    # Perform stepweight decay
                    p.data.mul_(1 - step_size * weight_decay)

                # p = p - lr * grad / denom
                p.data.addcdiv_(grad, denom, value=-step_size)

        return loss
