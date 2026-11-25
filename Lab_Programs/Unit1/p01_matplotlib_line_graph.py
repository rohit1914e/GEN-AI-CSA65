"""Unit 1 - P1: Install Matplotlib and display a simple line graph."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

banner("P1 - Simple line graph with Matplotlib " + matplotlib.__version__)
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
marks = [62, 68, 71, 75, 82, 88]
plt.figure(figsize=(6, 3.5))
plt.plot(months, marks, marker="o", color="#2563eb", linewidth=2)
plt.title("Monthly test score"); plt.xlabel("Month"); plt.ylabel("Marks"); plt.grid(alpha=0.3)
path = os.path.join(OUT, "U1_P01_line_graph.png")
plt.tight_layout(); plt.savefig(path, dpi=120)
print("Data:", dict(zip(months, marks)))
print("Line graph saved to", path)
