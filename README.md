# Newton-Muon Workspace

This folder contains a local Newton-Muon implementation plus upstream references.

## Quick Start

```bash
cd ~/newton-muon
source .venv/bin/activate
PYTHONPATH=. python experiments/toy_quadratic.py --steps 200
PYTHONPATH=. python -m pytest tests
```

To rebuild the environment:

```bash
cd ~/newton-muon
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Contents

- `newton_muon/`: reusable PyTorch `Muon` and `NewtonMuon` optimizers.
- `experiments/toy_quadratic.py`: fast anisotropic linear regression benchmark.
- `experiments/cifar10_vit_smoke.py`: CIFAR-10 tiny ViT generalization smoke test.
- `experiments/analyze_metrics.py`: CSV summaries and loss/time plots.
- `modded-nanogpt/`: upstream Modded-NanoGPT clone.
- `Newton-Muon-reference/`: authors' reference code for arXiv 2604.01472.
- `reports/WORKLOG.md`: setup notes, constraints, and suggested runs.

The full Record #4 GPT-2 reproduction is not suitable for the local RTX 4070 12 GB card; the reference repo documents an 80 GB single-GPU requirement. Use the local scripts for correctness checks and smaller ablations on this machine.

## Current Smoke Results

- Toy anisotropic regression, target loss 2.0: AdamW hit target in 135 steps, Newton-Muon with `k=1` hit target in 171 steps, Muon did not hit target by 200 steps.
- CIFAR-10 TinyViT, 40 training steps: AdamW `1.9169`, Muon `1.8292`, Newton-Muon `1.8505`; Newton-Muon was close to Muon on loss but slightly slower.
