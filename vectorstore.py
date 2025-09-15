import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

DB_PATH = "db/faiss_index"

def build_vectorstore(documents, persist_directory=DB_PATH):

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)


    docs = [d for d in docs if isinstance(d.page_content, str) and d.page_content.strip()]


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(docs, embeddings)


    os.makedirs(persist_directory, exist_ok=True)
    with open(os.path.join(persist_directory, "faiss_index.pkl"), "wb") as f:
        pickle.dump(db, f)

    print(f"[+] Vectorstore gespeichert unter: {persist_directory}")
    return db

def load_vectorstore(persist_directory=DB_PATH):

    with open(os.path.join(persist_directory, "faiss_index.pkl"), "rb") as f:
        db = pickle.load(f)
    print(f"[+] Vectorstore geladen von: {persist_directory}")
    return db
