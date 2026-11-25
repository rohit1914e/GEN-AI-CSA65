"""run_all.py - runs every lab program and saves its real console output to outputs/<Unit>_<program>.txt.
Usage: python run_all.py [Unit1 Unit2 ...] [--only p01,p02]"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
units = [a for a in sys.argv[1:] if a.startswith("Unit")] or ["Unit1", "Unit2"]
only = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")), None)
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
for u in units:
    for f in sorted(os.listdir(os.path.join(HERE, u))):
        if not (f.startswith("p") and f.endswith(".py")) or (only and f[:3] not in only):
            continue
        t = time.time()
        r = subprocess.run([sys.executable, os.path.join(u, f)], cwd=HERE, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=900)
        out = r.stdout + (("\n[stderr]\n" + "\n".join(l for l in r.stderr.splitlines() if "Warning" not in l and l.strip())[-1500:]) if r.returncode else "")
        with open(os.path.join(HERE, "outputs", f"{u}_{f[:-3]}.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"$ python {u}/{f}\n{out}\n[exit code {r.returncode}, {time.time() - t:.1f}s]\n")
        print(f"{u}/{f}: exit {r.returncode} ({time.time() - t:.1f}s)", flush=True)
