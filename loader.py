import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document

def load_documents(vault_path: str):

    loader = DirectoryLoader(
        vault_path,
        glob="**/*.md",
        loader_cls=lambda p: TextLoader(p, encoding="utf-8"),
        show_progress=True
    )
    docs = loader.load()


    clean_docs = [d for d in docs if getattr(d, "page_content", "").strip()]
    print(f"[+] {len(clean_docs)} gültige Dokumente geladen")
    return clean_docs
