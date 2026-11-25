"""Unit 1 - P6: Create tensors using PyTorch and perform basic mathematical operations."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
import torch

banner("P6 - PyTorch tensors, torch " + torch.__version__)
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])
print("a =\n", a); print("b =\n", b)
print("a + b =\n", a + b)
print("a * b (element-wise) =\n", a * b)
print("a @ b (matrix product) =\n", a @ b)
print("mean(a) =", a.mean().item(), "| max(b) =", b.max().item())
x = torch.tensor(3.0, requires_grad=True); y = x ** 2 + 2 * x
y.backward(); print("Autograd: y = x^2 + 2x at x=3 -> dy/dx =", x.grad.item())
