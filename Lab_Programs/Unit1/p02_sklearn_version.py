"""Unit 1 - P2: Install Scikit-learn and print its version."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
import sklearn

banner("P2 - Scikit-learn version")
print("scikit-learn version:", sklearn.__version__)
