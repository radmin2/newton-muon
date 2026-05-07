import math

import torch
from torch import nn

from newton_muon import Muon, NewtonMuon


def test_hook_updates_covariance_and_inverse():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 4, bias=False))
    opt = NewtonMuon(model.parameters(), lr=1e-3, refresh_interval=1, max_samples=None)
    opt.attach(model)

    x = torch.randn(16, 8)
    y = model(x).square().mean()
    y.backward()
    opt.step()

    stats = opt.activation_stats()
    assert len(stats) == 1
    assert stats[0]["batches"] == 1
    assert stats[0]["refreshes"] == 1
    assert stats[0]["cholesky_failures"] == 0
    assert math.isfinite(stats[0]["trace"])


def test_newton_muon_smoke_decreases_linear_regression_loss():
    torch.manual_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Linear(16, 8, bias=False, device=device)
    target = torch.randn(8, 16, device=device)
    opt = NewtonMuon(model.parameters(), lr=0.015, refresh_interval=2, max_samples=None)
    opt.attach(model)

    x = torch.randn(128, 16, device=device)
    y = x @ target.T

    losses = []
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        loss = (model(x) - y).square().mean()
        losses.append(float(loss.detach().cpu()))
        loss.backward()
        opt.step()

    assert losses[-1] < losses[0]


def test_plain_muon_skips_non_matrix_parameters():
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    opt = Muon(model.parameters(), lr=1e-3)
    x = torch.randn(8, 4)
    loss = model(x).square().mean()
    loss.backward()
    opt.step()
