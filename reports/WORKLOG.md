# Newton-Muon Worklog

## Environment

- Workspace: `/home/r_2/newton-muon`
- GPU visible from WSL: NVIDIA GeForce RTX 4070, 12 GB
- Python env: `/home/r_2/newton-muon/.venv`
- PyTorch: CUDA build installed in the venv

## Repositories

- `modded-nanogpt`: cloned from `https://github.com/KellerJordan/modded-nanogpt.git`
- `Newton-Muon-reference`: cloned from `https://github.com/zhehangdu/Newton-Muon.git`
- Paper text extracted to `paper_2604.01472v1.txt`

## Paper Notes

Newton-Muon applies the standard Muon pipeline to a right-preconditioned gradient:

`G_pre = G (K + gamma I)^-1`, where `K ~= E[z z^T]`.

The paper's Record #4 settings are `beta=0.95`, `gamma=0.2`, refresh interval `k=32`, and Newton-Muon LR `0.0040`. The reference repository states that the full Record #4 reproduction requires a single GPU with at least 80 GB RAM, so RTX 4070 runs should be treated as smoke tests or scaled ablations.

## Local Implementation

- `newton_muon/optimizer.py` implements:
  - `Muon`
  - `NewtonMuon`
  - forward pre-hooks for `nn.Linear` activation second moments
  - FP32 EWMA covariance buffers
  - Cholesky inverse with trace-scaled ridge and retry fallback
  - preconditioning before momentum and Newton-Schulz

## Suggested Runs

```bash
cd ~/newton-muon
source .venv/bin/activate
PYTHONPATH=. python experiments/toy_quadratic.py --steps 200
PYTHONPATH=. python experiments/cifar10_vit_smoke.py --optimizer newton_muon --steps 200
PYTHONPATH=. python experiments/analyze_metrics.py reports/toy_quadratic_metrics.csv --target-loss 2.0
```

For full GPT-2 experiments, use the reference scripts only on an 80 GB GPU:

```bash
cd ~/newton-muon/Newton-Muon-reference
source ../.venv/bin/activate
python data/cached_fineweb10B.py 50
python train_gpt_muon_1.py
python train_gpt_newton_muon_1.py
```

## First RTX 4070 Toy Results

On `experiments/toy_quadratic.py --steps 200 --d-in 128 --d-out 128 --condition 64`, AdamW is the strongest baseline. Newton-Muon is sensitive to refresh interval: with `lr=0.001`, `k=1` reached about `1.43` loss, while `k=8` reached about `24.45`, and Muon reached about `10.95`. This is a useful negative/sensitivity result for non-LLM toy structure, not evidence of a GPT-2 result.

For target loss `2.0` on the same toy task:

| Optimizer | Steps to target | Final loss | Wall time |
| --- | ---: | ---: | ---: |
| AdamW | 135 | 0.3275 | 0.171 s |
| Muon | did not hit | 10.9525 | 0.395 s |
| Newton-Muon (`k=1`) | 171 | 1.4260 | 0.442 s |

## First CIFAR-10 TinyViT Smoke

Command pattern:

```bash
PYTHONPATH=. python experiments/cifar10_vit_smoke.py --optimizer adamw --steps 40 --batch-size 128 --out reports/cifar10_vit_smoke_metrics.csv
PYTHONPATH=. python experiments/cifar10_vit_smoke.py --optimizer muon --steps 40 --batch-size 128 --out reports/cifar10_vit_smoke_metrics.csv
PYTHONPATH=. python experiments/cifar10_vit_smoke.py --optimizer newton_muon --steps 40 --batch-size 128 --out reports/cifar10_vit_smoke_metrics.csv
```

| Optimizer | Final loss | Wall time | Throughput |
| --- | ---: | ---: | ---: |
| AdamW | 1.9169 | 0.850 s | 6021 img/s |
| Muon | 1.8292 | 1.303 s | 3929 img/s |
| Newton-Muon | 1.8505 | 1.326 s | 3862 img/s |

This is only a short smoke test. It suggests Newton-Muon generalizes as executable code to a ViT-style workload, but its current generic PyTorch hook implementation has measurable overhead and did not beat Muon in 40 steps.
