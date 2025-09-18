import os
from dotenv import load_dotenv
from loader import load_documents
from vectorstore import build_vectorstore, load_vectorstore
from qa import GroqQA

from rich.console import Console
from rich.markdown import Markdown

os.environ["CUDA_VISIBLE_DEVICES"] = ""

console = Console()

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

    console.print("Obsidian-RAG-Bot mit Groq (quit mit 'exit')\n", style="bold green")
    while True:
        query = input("Frage: ")
        if query.lower().strip() in ["exit", "quit"]:
            break

        result = qa.ask(query)
        answer_md = result.get("answer", "")


        console.print(Markdown(answer_md))


        if result.get("source_documents"):
            console.print("\n📂 Quellen:", style="bold yellow")
            for doc in result["source_documents"]:
                console.print(f"- {doc.metadata.get('source')}")
        console.print("\n")

if __name__ == "__main__":
    chat()
