"""Unit 4 - P1: AI chatbot that answers student queries about an engineering college using a pre-trained language model
(FAQ retrieval with embeddings + Google Gemini for the reply)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, embed, load_text, gemini  # noqa: E402

banner("U4 P1 - Engineering college chatbot")
faqs = [f for f in load_text("college_faq.txt").split("\n\n") if f.strip()]
F = embed(faqs)


def chat(q):
    sims = F @ embed([q], query=True)[0]
    top = sims.argsort()[::-1][:3]
    if sims[top[0]] < 0.55:
        return "I can only help with college-related questions (fees, exams, hostel, library, placements...)."
    ctx = "\n".join(faqs[i] for i in top)
    return gemini(f"You are a friendly college helpdesk bot. Use only these FAQs.\n{ctx}\nStudent: {q}", temperature=0.3)


for q in ["hi, what time does the library close on saturday?", "can i pay my fees in two parts?",
          "what cgpa do I need to sit for placements?", "suggest a good movie for tonight"]:
    print(f"\nStudent: {q}\nBot    : {chat(q)}")
