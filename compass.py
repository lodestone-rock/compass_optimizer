import torch
from torch.optim import Optimizer


class Compass(Optimizer):
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
        super(Compass, self).__init__(params, defaults)

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
                    state["ema"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(p.data)

                ema, ema_squared = state["ema"], state["ema_squared"]
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
                    p.data.mul_(1 - step_size * weight_decay)

                # p = p - lr * grad / denom
                p.data.addcdiv_(grad, denom, value=-step_size)

        return loss


def hammer(train_steps, loss, amp_str=1):
    '''
    amplify the loss landscape steepness to make optimization problem trivial
    '''

    with torch.no_grad():
        loss_scaler = amp_str / loss
        loss_scaler = min(train_steps, loss_scaler)

    return loss * loss_scaler
