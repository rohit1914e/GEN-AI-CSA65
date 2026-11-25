"""Unit 3 - P10: Document-based AI assistant that supports multiple uploaded documents (PDF, DOCX, TXT) and question answering.
Usage: python p10_multi_document_assistant.py [file1.pdf file2.docx ...]   (demo files are created if none are given)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, embed, chunk_text, gemini, load_text, OUT  # noqa: E402
import faiss


def make_demo_files():
    """Create one PDF, one DOCX and one TXT so the demo has three formats."""
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    import docx
    pdf = os.path.join(OUT, "U3_P10_product_manuals.pdf"); st = getSampleStyleSheet()
    SimpleDocTemplate(pdf).build([Paragraph(p, st["BodyText"]) for p in load_text("product_manuals.txt").split("\n\n")])
    dx = os.path.join(OUT, "U3_P10_hr_policy.docx"); d = docx.Document()
    for p in load_text("hr_policy.txt").split("\n\n"):
        d.add_paragraph(p)
    d.save(dx)
    txt = os.path.join(OUT, "U3_P10_troubleshooting.txt")
    open(txt, "w", encoding="utf-8").write(load_text("engineering_troubleshooting.txt"))
    return [pdf, dx, txt]


def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return " ".join(p.extract_text() or "" for p in PdfReader(path).pages)
    if ext == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    if ext in (".txt", ".md"):
        return open(path, encoding="utf-8").read()
    raise ValueError(f"Unsupported file type: {ext}")


banner("U3 P10 - Multi-document assistant (PDF + DOCX + TXT)")
files = sys.argv[1:] or make_demo_files()
chunks = []
for f in files:
    try:
        text = read_any(f)
        chunks += [(os.path.basename(f), c) for c in chunk_text(text, 400, 60)]
        print(f"Loaded {os.path.basename(f):32s} {len(text):6d} chars")
    except Exception as e:
        print(f"Skipped {f}: {e}")
index = faiss.IndexFlatIP(384); index.add(embed([c for _, c in chunks]))
for q in ["How many sick leaves do employees get?", "What does washing machine error E10 mean?",
          "How to fix porosity in robotic welds?", "Compare the notice period with the laptop battery tip"]:
    s, ids = index.search(embed([q], query=True), 4)
    ctx = "\n".join(f"[{chunks[i][0]}] {chunks[i][1]}" for i in ids[0])
    print(f"\nQ: {q}\nA: " + gemini(f"Answer from the documents, cite [file]; say if something is not covered.\n{ctx}\nQ: {q}", temperature=0.1))
