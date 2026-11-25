"""Unit 3 - P8: Context-aware chatbot using LangChain, retrieval and an LLM (remembers the conversation)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, embed, load_text, chunk_text, gemini  # noqa: E402
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS


class LocalEmbeddings(Embeddings):
    """LangChain wrapper around the local bge-small embedding model."""
    def embed_documents(self, texts):
        return embed(texts).tolist()

    def embed_query(self, text):
        return embed([text], query=True)[0].tolist()


llm = RunnableLambda(lambda pv: gemini(pv.to_string(), temperature=0.2))   # Gemini as a LangChain runnable
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

banner("U3 P8 - Context-aware chatbot (chat history + retrieval)")
docs = [Document(page_content=c) for c in chunk_text(load_text("os_memory_notes.txt"), 350, 70)]
retriever = FAISS.from_documents(docs, LocalEmbeddings()).as_retriever(search_kwargs={"k": 2})
rewrite = ChatPromptTemplate.from_messages([MessagesPlaceholder("history"), ("human",
    "Rewrite my last question as a standalone question (resolve words like 'it', 'that'): {question}. Return only the question.")])
answer = ChatPromptTemplate.from_messages([("system", "Answer from the context in 2-3 sentences.\nContext:\n{context}"),
                                           MessagesPlaceholder("history"), ("human", "{question}")])
history = []
for q in ["What is paging?", "What problem does it remove?", "And which replacement algorithm suffers from Belady's anomaly?"]:
    standalone = (rewrite | llm | StrOutputParser()).invoke({"history": history, "question": q}).strip()
    ctx = "\n".join(d.page_content for d in retriever.invoke(standalone))
    a = (answer | llm | StrOutputParser()).invoke({"history": history, "question": q, "context": ctx})
    print(f"\nUser: {q}\n  (standalone: {standalone})\nBot : {a}")
    history += [HumanMessage(q), AIMessage(a)]
