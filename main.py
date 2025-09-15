import os
from dotenv import load_dotenv
from loader import load_documents
from vectorstore import build_vectorstore, load_vectorstore
from qa import GroqQA

os.environ["CUDA_VISIBLE_DEVICES"] = ""


load_dotenv()

VAULT_PATH = "/home/matti/Obsidian"
DB_PATH = "db/faiss_index"

def init_db():

    documents = load_documents(VAULT_PATH)
    documents = [doc for doc in documents if "Templates/" not in doc.metadata.get("source", "")]
    print(f"[+] Geladene Dokumente (nach Filter): {len(documents)}")

    if not os.path.exists(os.path.join(DB_PATH, "faiss_index.pkl")):
        db = build_vectorstore(documents, persist_directory=DB_PATH)
    else:
        db = load_vectorstore(persist_directory=DB_PATH)

    return db

def chat():
    db = init_db()
    qa = GroqQA(db)

    print("Obsidian-RAG-Bot mit Groq (quit mit 'exit')\n")
    while True:
        query = input("Frage: ")
        if query.lower().strip() in ["exit", "quit"]:
            break

        print("\n", end=" ", flush=True)
        result = qa.ask(query)
        print("\n")

        if hasattr(result, "source_documents") and result.source_documents:
            print("\n📂 Quellen:")
            for doc in result.source_documents:
                print("-", doc.metadata.get("source"))
        print("\n")


if __name__ == "__main__":
    chat()
