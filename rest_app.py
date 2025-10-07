from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from flask_cors import CORS

from loader import load_documents
from vectorstore import build_vectorstore, load_vectorstore
from qa import GroqQA

os.environ["CUDA_VISIBLE_DEVICES"] = ""

load_dotenv()

VAULT_PATH = "/home/matti/Obsidian"
DB_PATH = "db/faiss_index"

app = Flask(__name__)
CORS(app)

def init_db():
    documents = load_documents(VAULT_PATH)
    documents = [doc for doc in documents if "Templates/" not in doc.metadata.get("source", "")]
    print(f"[+] Geladene Dokumente (nach Filter): {len(documents)}")

    if not os.path.exists(os.path.join(DB_PATH, "faiss_index.pkl")):
        db = build_vectorstore(documents, persist_directory=DB_PATH)
    else:
        db = load_vectorstore(persist_directory=DB_PATH)

    return db

print("[*] Initialisiere DB...")
db = init_db()
qa = GroqQA(db)
print("[*] Bot bereit!")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Keine Frage gesendet"}), 400

    result = qa.ask(query)
    response = {
        "answer": result.get("answer", ""),
        "source_documents": [
            doc.metadata.get("source") for doc in result.get("source_documents", [])
        ]
    }
    return jsonify(response)


@app.route("/msg")
def msg():
    return {"msg": "Hello, World!"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
