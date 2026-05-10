from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn

# =====================================================================
# 1. КОМПИЛИРУЕМЫЕ ЯДРА (Вынесены из классов для кэширования графов)
# =====================================================================

@torch.compile
def zeropower_via_newtonschulz5(g: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Оптимизированный Newton-Schulz без вызовов .contiguous() и копирования памяти."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16() if g.is_cuda else g.float()
    x = x / (x.norm(dim=(-2, -1), keepdim=True).clamp_min(eps))

    m, n = x.shape[-2], x.shape[-1]
    
    # Динамический выбор пути (как в V1), чтобы избежать аллокаций
    if m > n:
        for _ in range(steps):
            A = x.transpose(-2, -1) @ x
            B = b * A + c * (A @ A)
            x = a * x + x @ B
    else:
        for _ in range(steps):
            A = x @ x.transpose(-2, -1)
            B = b * A + c * (A @ A)
            x = a * x + B @ x

    return x.to(dtype=g.dtype)


@torch.compile
def _fused_update_cov(x_sliced: Tensor, cov: Tensor, beta: float, batches: Tensor) -> None:
    """Слитая математика обновления ковариации. Выполняется на GPU без Python-оверхеда."""
    # Перевод во float32 делается ТОЛЬКО для уже обрезанного тензора
    x_f32 = x_sliced.to(torch.float32)
    # Матричное умножение и нормализация
    batch_cov = (x_f32.T @ x_f32) * (1.0 / x_f32.shape[0])
    # Плавное обновление EMA
    cov.lerp_(batch_cov, 1.0 - beta)
    batches.add_(1)


# =====================================================================
# 2. БАЗОВЫЙ КЛАСС MUON (Оставлен чистым)
# =====================================================================

class Muon(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1e-3, momentum: float = 0.95,
                 weight_decay: float = 0.0, nesterov: bool = True, ns_steps: int = 5,
                 scale_updates: bool = True) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, ns_steps=ns_steps, scale_updates=scale_updates)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.grad.ndim != 2:
                    continue

                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                g = p.grad.detach().float()
                state = self.state[p]
                
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(g)
                    state["momentum_buffer"] = buf
                    
                buf.mul_(group["momentum"]).add_(g)
                update = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
                if group["scale_updates"]:
                    update.mul_(max(update.shape[-2], update.shape[-1]) ** 0.5)
                    
                p.add_(update.to(dtype=p.dtype), alpha=-group["lr"])

        return loss

# =====================================================================
# 3. ОПТИМИЗИРОВАННЫЙ NEWTON-MUON
# =====================================================================

class NewtonMuon(Muon):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1e-3, momentum: float = 0.95,
                 weight_decay: float = 0.0, nesterov: bool = True, ns_steps: int = 5,
                 scale_updates: bool = True, beta: float = 0.95, ridge: float = 0.2,
                 refresh_interval: int = 32, init_scale: float = 1e-3, eps: float = 1e-8,
                 max_samples: int | None = 2048, # Снижено до 2048 для скорости без потери качества
                 cholesky_retries: int = 5, bias_correction: bool = True) -> None:
        
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay,
                         nesterov=nesterov, ns_steps=ns_steps, scale_updates=scale_updates)
        
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
        self._handles: list[Any] =[]
        self._managed_params: list[nn.Parameter] =[]

    def attach(self, model: nn.Module, module_filter: Callable[[str, nn.Linear], bool] | None = None) -> "NewtonMuon":
        self.remove_hooks()
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or id(module.weight) not in self._param_ids:
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
            state["activation_cov"] = torch.eye(d, dtype=torch.float32, device=p.device).mul_(self.init_scale)
            state["activation_inv"] = torch.eye(d, dtype=torch.float32, device=p.device)
            state["activation_batches"] = torch.zeros((), dtype=torch.long, device=p.device)
            state["activation_refreshes"] = torch.zeros((), dtype=torch.long, device=p.device)
            state["cholesky_failures"] = torch.zeros((), dtype=torch.long, device=p.device)

    def _make_activation_hook(self, p: nn.Parameter):
        # ЗАМЫКАНИЕ: Достаем тензоры ИЗ СЛОВАРЯ ОДИН РАЗ при создании хука.
        # Это полностью убирает Python-поиск по словарю `self.state` в forward проходе!
        state = self.state[p]
        cov_tensor = state["activation_cov"]
        batches_tensor = state["activation_batches"]
        
        beta = self.beta
        max_samples = self.max_samples

        @torch.no_grad()
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs: return
            x = inputs[0]
            if not torch.is_tensor(x) or x.shape[-1] != p.shape[1] or not x.is_floating_point():
                return

            x2d = x.view(-1, x.shape[-1])
            if x2d.numel() == 0: return
            
            # Subsampling делается ОЧЕНЬ ДЕШЕВО (через stride, без копирования памяти) 
            # ДО того, как мы конвертируем тензор во float32
            if max_samples is not None and x2d.shape[0] > max_samples:
                stride = max(1, x2d.shape[0] // max_samples)
                x2d = x2d[::stride][:max_samples]

            # Вся тяжелая математика отдается в скомпилированное ядро
            _fused_update_cov(x2d, cov_tensor, beta, batches_tensor)

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

            k = cov.clone() # nan_to_num убрано, так как _fused_update_cov надежен
            batches = int(state["activation_batches"].item())
            
            if self.bias_correction and batches > 0:
                k.div_(max(1.0 - self.beta**batches, self.eps))
                
            d = k.shape[0]
            eye = torch.eye(d, dtype=k.dtype, device=k.device)
            base_ridge = (k.diagonal().sum() / float(d)).clamp_min(0.0) * self.ridge + self.eps

            ok = False
            for retry in range(max(1, self.cholesky_retries)):
                # Быстрое слияние сложения и умножения
                candidate = torch.addcmul(k, eye, base_ridge * (10.0**retry))
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
        loss = closure() if closure is not None else None
        self.refresh_preconditioners()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.grad.ndim != 2:
                    continue

                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                g = p.grad.detach().float()
                inv = self.state[p].get("activation_inv")
                
                # Применение предобуславливателя Newton-Muon
                if inv is not None and g.shape[-1] == inv.shape[0]:
                    g = g @ inv # Быстрое нативное перемножение

                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(g)
                    state["momentum_buffer"] = buf
                    
                buf.mul_(group["momentum"]).add_(g)
                update = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                # Обращение к скомпилированному ядру
                update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
                if group["scale_updates"]:
                    update.mul_(max(update.shape[-2], update.shape[-1]) ** 0.5)
                    
                p.add_(update.to(dtype=p.dtype), alpha=-group["lr"])

        self._step += 1
        return loss