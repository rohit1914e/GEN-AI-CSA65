"""shots.py - builds zero-shot, one-shot and few-shot prompts for a task (used by Unit 2 P1-P5)."""


def build(task, examples):
    """Return {'zero-shot': p, 'one-shot': p, 'few-shot': p}. examples = [(input, output), ...]."""
    ex = lambda n: "\n\n".join(f"Example {i}:\nInput: {a}\nOutput: {b}" for i, (a, b) in enumerate(examples[:n], 1))
    return {"zero-shot": task,
            "one-shot": f"{ex(1)}\n\nNow do the same.\n{task}",
            "few-shot": f"{ex(len(examples))}\n\nFollow the style of the examples.\n{task}"}
