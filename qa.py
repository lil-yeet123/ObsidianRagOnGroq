import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class GroqQA:
    def __init__(self, vectorstore, model="openai/gpt-oss-20b"):

        self.retriever = vectorstore.as_retriever()
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def ask(self, question: str, stream=True, **kwargs):
        docs = self.retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"📂 Kontext:\n{context}\n\n❓ Frage:\n{question}"

        messages = [
            {
                "role": "system",
                "content": """Du bist ein persönlicher Assistent mit Zugriff auf ein Obsidian-Vault.
Regeln:
- Nutze ausschließlich die Kontextinformationen aus dem Vault
- Falls unzureichend: sag es offen
- Antworte klar, präzise, strukturiert
- Sprache = Sprache der Frage
- Quellen angeben, falls möglich
- Allgemeinwissen nur kennzeichnen
- Keine Markdown-Ausgabe, außer explizit verlangt"""
            },
            {"role": "user", "content": prompt}
        ]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            **kwargs
        )


        text = ""
        for chunk in completion:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                print(delta.content, end="", flush=True)
                text += delta.content
        print()


        result = {
            "answer": text,
            "source_documents": docs
        }

        return result
