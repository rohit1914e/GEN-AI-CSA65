"""Unit 5 - P3: Integrate a locally running LLM with a Python application to implement a question-answering system
(the app supplies the student-handbook text as context; the local LLM answers)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, local_llm, load_text  # noqa: E402

banner("U5 P3 - Question answering app with a local LLM")
CONTEXT = load_text("student_handbook.txt")


def answer(q):
    return local_llm(f"Answer the question using ONLY the handbook below. If it is not there, say 'Not in the handbook'.\n\n"
                     f"HANDBOOK:\n{CONTEXT}\n\nQUESTION: {q}\nANSWER (1-2 sentences):", temperature=0)


for q in ["What is the minimum attendance?", "How many books can I borrow from the library?", "Are smart watches allowed in exams?",
          "What is the canteen menu?"]:
    print(f"\nQ: {q}\nA: {answer(q)}")
