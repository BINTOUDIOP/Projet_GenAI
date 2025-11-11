import os, glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
DB_DIR = os.getenv("CHROMA_DB_DIR", ".chroma")

def load_docs(data_dir="data"):
    docs = []
    for path in glob.glob(os.path.join(data_dir, "**/*"), recursive=True):
        if path.lower().endswith(".pdf"):
            docs += PyPDFLoader(path).load()
        elif path.lower().endswith(".docx"):
            docs += Docx2txtLoader(path).load()
        elif path.lower().endswith((".txt", ".md")):
            docs += TextLoader(path, encoding="utf-8").load()
    return docs

def main():
    docs = load_docs()
    if not docs:
        print("Aucun document trouvé dans ./data")
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDINGS_MODEL","text-embedding-3-small"))
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="corp_doc"
    )
    print(f"✅ Index construit avec {len(chunks)} chunks → {DB_DIR}")

if __name__ == "__main__":
    main()