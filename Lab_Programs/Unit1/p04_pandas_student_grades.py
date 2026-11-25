"""Unit 1 - P4: Create a Pandas DataFrame with student names and grades."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
import pandas as pd

banner("P4 - Pandas DataFrame of students and grades")
df = pd.DataFrame({"Name": ["Rohit", "Asha", "Bala", "Divya", "Karthik"],
                   "Marks": [91, 78, 66, 85, 54]})
df["Grade"] = pd.cut(df["Marks"], bins=[0, 49, 59, 69, 79, 89, 100], labels=["RA", "C", "B", "B+", "A", "O"])
print(df.to_string(index=False))
print("\nClass average:", df["Marks"].mean())
print("Top student :", df.loc[df["Marks"].idxmax(), "Name"])
