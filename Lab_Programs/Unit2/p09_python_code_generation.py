"""Unit 2 - P9: Python Code Generation using Structured Prompting - generate programs and validate the generated code."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, gemini  # noqa: E402
import re

banner("U2 P9 - Structured prompt -> Python code -> automatic validation")
TEMPLATE = ("ROLE: Senior Python developer.\nTASK: Write a Python function named {name} that {desc}\n"
            "CONSTRAINTS: standard library only, include a docstring, no input(), no printing.\n"
            "OUTPUT FORMAT: only one python code block.")
problems = [("is_prime(n)", "returns True if n is a prime number", [(2, True), (15, False), (97, True), (1, False)]),
            ("fibonacci(n)", "returns the n-th Fibonacci number with fibonacci(0)=0 and fibonacci(1)=1", [(0, 0), (1, 1), (10, 55)]),
            ("reverse_words(s)", "returns the sentence with the order of words reversed", [("I love AI", "AI love I")])]
for name, desc, tests in problems:
    code = gemini(TEMPLATE.format(name=name, desc=desc), temperature=0.2)
    code = re.search(r"```(?:python)?\n(.*?)```", code, re.S).group(1) if "```" in code else code
    print(f"\n### {name}\n{code.strip()}")
    ns = {}
    try:
        exec(compile(code, name, "exec"), ns)
        fn = ns[name.split("(")[0]]
        results = [(arg, exp, fn(arg)) for arg, exp in tests]
        ok = all(e == g for _, e, g in results)
        print("Validation:", "PASSED" if ok else "FAILED", [(a, g) for a, _, g in results])
    except Exception as e:
        print("Validation: ERROR", type(e).__name__, e)
