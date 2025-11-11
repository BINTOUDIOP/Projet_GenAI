import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

load_dotenv()
DB_DIR = os.getenv("CHROMA_DB_DIR", ".chroma")

def get_retriever(k=4):
    vectordb = Chroma(
        persist_directory=DB_DIR,
        collection_name="corp_doc",
        embedding_function=OpenAIEmbeddings(model=os.getenv("EMBEDDINGS_MODEL","text-embedding-3-small"))
    )
    return vectordb.as_retriever(search_kwargs={"k": k})

SYSTEM = """Tu es un assistant qui répond en citant le CONTENU des documents.
- Si l'info n'est pas dans les extraits, dis-le honnêtement.
- Réponds en français, concis, avec puces si utile.
Inclure une courte section "Sources" avec les noms de fichiers pertinents si possible."""
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Question: {question}\n\nExtraits:\n{context}")
])

def format_docs(docs):
    out = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "document")
        out.append(f"[{src}] {d.page_content}")
    return "\n\n---\n\n".join(out[:8])

def build_rag_chain():
    llm = ChatOpenAI(model=os.getenv("CHAT_MODEL","gpt-4o-mini"), temperature=0.2)
    retriever = get_retriever()

    chain = RunnableMap({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }) | PROMPT | llm
    return chain

def answer(question: str):
    chain = build_rag_chain()
    return chain.invoke(question).content