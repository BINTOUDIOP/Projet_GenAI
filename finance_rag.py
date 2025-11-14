import os
from dotenv import load_dotenv
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.tools import tool


load_dotenv()
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "./data/finance_document.pdf")
PERSIST_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db_agent")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialisation des composants
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")




def format_docs(docs: List[Document]) -> str:
    """Formate les documents (chunks) récupérés pour insertion dans le prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_retriever(documents_path: str = DOCUMENT_PATH, persist_dir: str = PERSIST_DIR):
    """
    Crée l'index vectoriel ChromaDB ou le charge s'il existe.
    """
    if not os.path.exists(documents_path):
        print(f" Erreur RAG : Document non trouvé à {documents_path}. Indexation annulée.")
        return None


    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"Chargement de l'index Chroma existant depuis {persist_dir}...")
        try:
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            print("Index Chroma chargé.")
            return vectorstore.as_retriever(k=5)
        except Exception as e:
            print(f" Erreur de chargement Chroma: {e}")
            pass


    print(f" Début de l'ingestion du document : {documents_path}")

    # A. Chargement
    loader = PyPDFLoader(documents_path)
    docs = loader.load()

    # B. Fractionnement
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(docs)

    # C. Indexation (Chroma)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f" Indexation ChromaDB terminée. {len(splits)} chunks indexés.")
    # D. Création du Retriever
    return vectorstore.as_retriever(k=5)


# Création du Pipeline RAG LCEL

# 1. Prompt RAG
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Vous êtes un assistant financier expert. Répondez à la question en utilisant UNIQUEMENT le contexte financier interne fourni ci-dessous. Si la réponse n'est pas dans le contexte, dites poliment que vous ne possédez pas cette information dans vos documents internes."),
    ("human", "Contexte: {context}\n\nQuestion: {question}"),
])


# 2. Fonction de construction de la chaîne
def create_rag_chain(retriever):
    """Construit la chaîne RAG avec LCEL."""
    rag_chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | llm
            | StrOutputParser()
    )
    return rag_chain


# Exécuté à l'importation
FINANCE_RETRIEVER = create_retriever()
RAG_CHAIN = create_rag_chain(FINANCE_RETRIEVER) if FINANCE_RETRIEVER else None

@tool
def run_rag_tool(question: str) -> str:
    """
    recherche_interne
    UTILISEZ CET OUTIL UNIQUEMENT pour répondre à des questions sur les MANUELS, les RAPPORTS ou la THÉORIE FINANCIÈRE contenus dans les documents internes.
    """
    if RAG_CHAIN:
        return RAG_CHAIN.invoke(question)
    return "Erreur : Le système de documents internes n'est pas disponible (index non chargé)."