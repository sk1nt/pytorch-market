PyTorch installation
====================

This project uses PyTorch as an optional backend for neural-network models. Below are minimal instructions for installing PyTorch for CPU-only and common CUDA-enabled setups.

CPU-only (recommended for CI and quick development)
----------------------------------------------------

1. Upgrade pip:

```bash
python -m pip install --upgrade pip
```

2. Install the CPU-only wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

CUDA-enabled (choose the wheel matching your CUDA version)
---------------------------------------------------------

Example for CUDA 12.1:

```bash
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Example for CUDA 11.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Notes
-----
- If you need GPU acceleration, pick the wheel that matches your system's CUDA toolkit / driver compatibility. See the official PyTorch site for guidance: https://pytorch.org/get-started/locally/
- The package name on PyPI is ``torch`` (not ``pytorch``), which is the correct entry added to ``requirements.txt``.

Verify installation
-------------------

In Python run:

```python
import torch
print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
```

If ``torch`` imports successfully and prints a version, installation is correct.
