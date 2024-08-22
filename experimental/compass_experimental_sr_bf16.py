import torch
from torch.optim import Optimizer

# @torch.compile
def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    # thanks to Nerogar for fast stochastic pytorch implementation 
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    with torch.no_grad():
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))


class Compasstic(Optimizer):
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
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
        )
        super(Compasstic, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                assert p.dtype == torch.bfloat16, "only bfloat 16 is supported."
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["ema"] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(p.data, dtype=torch.bfloat16)

                # unpack
                grad = grad.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)
                ema = state["ema"].to(torch.float32)
                ema_squared = state["ema_squared"].to(torch.float32)

                beta1, beta2 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                state["step"] += 1

                # center the gradient vector
                if centralization != 0:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # bias correction step size
                # soft warmup
                bias_correction = 1 - beta1 ** state["step"]
                bias_correction_sqrt = (1 - beta2 ** state["step"]) ** (1 / 2)
                step_size = lr / bias_correction

                # Decay the first and second moment running average coefficient
                # ema = ema + (1 - beta1) * grad
                ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amplification_factor
                grad.add_(ema, alpha=amplification_factor)
                # ema_squared = ema + (1 - beta2) * grad ** 2
                ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])

                if weight_decay != 0:
                    # Perform stepweight decay
                    p_fp32.data.mul_(1 - step_size * weight_decay)

                # p = p - lr * grad / denom
                p_fp32.data.addcdiv_(grad, denom, value=-step_size)

                # pack
                copy_stochastic_(state["ema"], ema)
                copy_stochastic_(state["ema_squared"], ema_squared)
                copy_stochastic_(p, p_fp32) 

        return loss


import torch
from torch.optim import Optimizer

# @torch.compile
def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    # thanks to Nerogar for fast stochastic pytorch implementation 
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    with torch.no_grad():
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))


class LPFAdamW(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.9, 0.999)).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.9, 0.999),
        amp_fac=2,
        eps=1e-8,
        weight_decay=0,
        centralization=0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
        )
        super(LPFAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                assert p.dtype == torch.bfloat16, "only bfloat 16 is supported."
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["smoothing"] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    state["ema"] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(p.data, dtype=torch.bfloat16)

                # unpack
                grad = grad.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)
                smoothing = state["smoothing"].to(torch.float32)
                ema = state["ema"].to(torch.float32)
                ema_squared = state["ema_squared"].to(torch.float32)

                beta1, beta2, beta3 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                state["step"] += 1

                # center the gradient vector
                if centralization != 0:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # bias correction step size
                # soft warmup
                bias_correction = 1 - beta2 ** state["step"]
                bias_correction_sqrt = (1 - beta3 ** state["step"]) ** (1 / 2)
                step_size = lr / bias_correction

                # Decay the first and second moment running average coefficient
                # ema = ema + (1 - beta2) * grad
                smoothing.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amplification_factor
                grad.add_(smoothing, alpha=amplification_factor)

                ema.mul_(beta2).add_(grad, alpha=1 - beta2)
                # ema_squared = ema + (1 - beta3) * grad ** 2
                ema_squared.mul_(beta3).addcmul_(grad, grad, value=1 - beta3)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])

                if weight_decay != 0:
                    # Perform stepweight decay
                    p_fp32.data.mul_(1 - step_size * weight_decay)

                # p = p - lr * grad / denom
                p_fp32.data.addcdiv_(ema, denom, value=-step_size)

                # pack
                copy_stochastic_(state["ema"], ema)
                copy_stochastic_(state["ema_squared"], ema_squared)
                copy_stochastic_(state["smoothing"], smoothing)
                copy_stochastic_(p, p_fp32) 

        return loss
