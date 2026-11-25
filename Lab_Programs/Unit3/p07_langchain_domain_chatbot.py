"""Unit 3 - P7: Domain-specific chatbot using LangChain and a vector database (HR policy bot, FAISS)."""
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

banner("U3 P7 - Domain-specific HR chatbot with LangChain + FAISS")
docs = [Document(page_content=c, metadata={"source": "hr_policy.txt"}) for c in chunk_text(load_text("hr_policy.txt"), 400, 60)]
store = FAISS.from_documents(docs, LocalEmbeddings())
retriever = store.as_retriever(search_kwargs={"k": 3})
prompt = ChatPromptTemplate.from_template("You are the company HR assistant. Answer ONLY from the policy context; "
                                          "if it is not covered say so.\nContext:\n{context}\n\nQuestion: {question}")
chain = ({"context": retriever | RunnableLambda(lambda ds: "\n".join(d.page_content for d in ds)),
          "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
for q in ["How many earned leaves can I carry forward?", "Can I work from home?", "What is the dress code for Fridays?"]:
    print(f"\nUser: {q}\nBot : {chain.invoke(q)}")
