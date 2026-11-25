"""Unit 1 - P3: Create a NumPy array containing the first 20 natural numbers."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
import numpy as np

banner("P3 - NumPy array of the first 20 natural numbers")
a = np.arange(1, 21)
print("Array :", a)
print("Shape :", a.shape, "| dtype:", a.dtype)
print("Sum =", a.sum(), "| Mean =", a.mean(), "| Even numbers:", a[a % 2 == 0])
print("Reshaped 4x5:"); print(a.reshape(4, 5))
