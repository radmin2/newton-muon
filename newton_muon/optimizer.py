from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn


def zeropower_via_newtonschulz5(g: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Muon's Newton-Schulz matrix-sign approximation for one matrix or a batch."""
    if g.ndim < 2:
        raise ValueError("Muon expects tensors with at least two matrix dimensions")

    a, b, c = (3.4445, -4.7750, 2.0315)
    work_dtype = torch.bfloat16 if g.is_cuda else torch.float32
    x = g.to(dtype=work_dtype)
    x = x / (x.norm(dim=(-2, -1), keepdim=True).clamp_min(eps))

    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    x = x.contiguous()

    for _ in range(steps):
        xx_t = x @ x.transpose(-2, -1)
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.transpose(-2, -1)
    return x.to(dtype=g.dtype)


class Muon(torch.optim.Optimizer):
    """Small, standalone Muon implementation for 2-D weight matrices."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        scale_updates: bool = True,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            scale_updates=scale_updates,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            scale_updates = group["scale_updates"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim != 2:
                    continue

                if weight_decay:
                    p.mul_(1.0 - lr * weight_decay)

                g = p.grad.detach().float()
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(g)
                    state["momentum_buffer"] = buf
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf

                update = self._orthogonalize(update, ns_steps)
                if scale_updates:
                    update = update.mul(max(update.shape[-2], update.shape[-1]) ** 0.5)
                p.add_(update.to(dtype=p.dtype), alpha=-lr)

        return loss

    def _orthogonalize(self, update: Tensor, ns_steps: int) -> Tensor:
        if update.ndim == 2 and update.shape[0] == 3 * update.shape[1]:
            return torch.cat(
                [zeropower_via_newtonschulz5(part, steps=ns_steps) for part in update.chunk(3, dim=0)],
                dim=0,
            )
        return zeropower_via_newtonschulz5(update, steps=ns_steps)


class NewtonMuon(Muon):
    """
    Muon with right-preconditioning by an EWMA of activation second moments.

    The optimizer attaches forward pre-hooks to nn.Linear modules. For each
    managed weight W with shape [out_features, in_features], it maintains
    K ~= E[z z^T] in FP32 and periodically refreshes inv(K + gamma I).
    During step(), raw gradients are right-multiplied by the cached inverse
    before momentum and Newton-Schulz orthogonalization.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        scale_updates: bool = True,
        beta: float = 0.95,
        ridge: float = 0.2,
        refresh_interval: int = 32,
        init_scale: float = 1e-3,
        eps: float = 1e-8,
        max_samples: int | None = 8192,
        cholesky_retries: int = 5,
        bias_correction: bool = True,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            scale_updates=scale_updates,
        )
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if refresh_interval < 1:
            raise ValueError("refresh_interval must be >= 1")

        self.beta = float(beta)
        self.ridge = float(ridge)
        self.refresh_interval = int(refresh_interval)
        self.init_scale = float(init_scale)
        self.eps = float(eps)
        self.max_samples = max_samples
        self.cholesky_retries = int(cholesky_retries)
        self.bias_correction = bool(bias_correction)

        self._step = 0
        self._param_ids = {id(p) for group in self.param_groups for p in group["params"]}
        self._handles: list[Any] = []
        self._managed_params: list[nn.Parameter] = []

    def attach(
        self,
        model: nn.Module,
        module_filter: Callable[[str, nn.Linear], bool] | None = None,
    ) -> "NewtonMuon":
        """Attach activation hooks to Linear layers whose weight is optimized here."""
        self.remove_hooks()
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if id(module.weight) not in self._param_ids:
                continue
            if module.weight.ndim != 2:
                continue
            if module_filter is not None and not module_filter(name, module):
                continue

            self._init_preconditioner_state(module.weight)
            self._managed_params.append(module.weight)
            self._handles.append(module.register_forward_pre_hook(self._make_activation_hook(module.weight)))
        return self

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._managed_params.clear()

    def _init_preconditioner_state(self, p: nn.Parameter) -> None:
        d = int(p.shape[1])
        state = self.state[p]
        if "activation_cov" not in state:
            cov = torch.eye(d, dtype=torch.float32, device=p.device).mul_(self.init_scale)
            inv = torch.eye(d, dtype=torch.float32, device=p.device)
            state["activation_cov"] = cov
            state["activation_inv"] = inv
            state["activation_batches"] = torch.zeros((), dtype=torch.long, device=p.device)
            state["activation_refreshes"] = torch.zeros((), dtype=torch.long, device=p.device)
            state["cholesky_failures"] = torch.zeros((), dtype=torch.long, device=p.device)

    def _make_activation_hook(self, p: nn.Parameter):
        @torch.no_grad()
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            if x.shape[-1] != p.shape[1]:
                return
            if not torch.is_floating_point(x):
                return

            x2d = x.detach().reshape(-1, x.shape[-1])
            if x2d.numel() == 0:
                return
            if self.max_samples is not None and x2d.shape[0] > self.max_samples:
                stride = max(1, x2d.shape[0] // self.max_samples)
                x2d = x2d[::stride][: self.max_samples]

            x2d = torch.nan_to_num(x2d.float())
            batch_cov = x2d.transpose(0, 1).matmul(x2d).div_(float(x2d.shape[0]))

            state = self.state[p]
            state["activation_cov"].lerp_(batch_cov, 1.0 - self.beta)
            state["activation_batches"].add_(1)

        return hook

    @torch.no_grad()
    def refresh_preconditioners(self, force: bool = False) -> None:
        if not force and (self._step + 1) % self.refresh_interval != 0:
            return

        for p in self._managed_params:
            state = self.state[p]
            cov = state.get("activation_cov")
            inv = state.get("activation_inv")
            if cov is None or inv is None:
                continue

            k = torch.nan_to_num(cov.float().clone())
            batches = int(state["activation_batches"].item())
            if self.bias_correction and batches > 0:
                correction = max(1.0 - self.beta**batches, self.eps)
                k.div_(correction)
            d = k.shape[0]
            eye = torch.eye(d, dtype=k.dtype, device=k.device)
            base_ridge = (k.diagonal().sum() / float(d)).clamp_min(0.0) * self.ridge + self.eps

            ok = False
            for retry in range(max(1, self.cholesky_retries)):
                candidate = k + eye * (base_ridge * (10.0**retry))
                chol, info = torch.linalg.cholesky_ex(candidate, upper=False, check_errors=False)
                if int(info.item()) == 0:
                    inv.copy_(torch.cholesky_inverse(chol, upper=False))
                    ok = True
                    break

            if not ok:
                inv.copy_(eye)
                state["cholesky_failures"].add_(1)
            state["activation_refreshes"].add_(1)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.refresh_preconditioners()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            scale_updates = group["scale_updates"]

            for p in group["params"]:
                if p.grad is None or p.grad.ndim != 2:
                    continue

                if weight_decay:
                    p.mul_(1.0 - lr * weight_decay)

                g = p.grad.detach().float()
                inv = self.state[p].get("activation_inv")
                if inv is not None and g.shape[-1] == inv.shape[0]:
                    g = g.matmul(inv)

                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(g)
                    state["momentum_buffer"] = buf
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf

                update = self._orthogonalize(update, ns_steps)
                if scale_updates:
                    update = update.mul(max(update.shape[-2], update.shape[-1]) ** 0.5)
                p.add_(update.to(dtype=p.dtype), alpha=-lr)

        self._step += 1
        return loss

    def activation_stats(self) -> list[dict[str, int | float | tuple[int, ...]]]:
        rows = []
        for p in self._managed_params:
            state = self.state[p]
            cov = state["activation_cov"]
            rows.append(
                {
                    "shape": tuple(p.shape),
                    "cov_shape": tuple(cov.shape),
                    "batches": int(state["activation_batches"].item()),
                    "refreshes": int(state["activation_refreshes"].item()),
                    "cholesky_failures": int(state["cholesky_failures"].item()),
                    "trace": float(cov.diagonal().sum().item()),
                }
            )
        return rows

    def __del__(self) -> None:
        self.remove_hooks()
