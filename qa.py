import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class GroqQA:
    def __init__(self, vectorstore, model="openai/gpt-oss-20b"):
        self.retriever = vectorstore.as_retriever()
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model
        self.chat_history = []
        self.system_prompt = {
            "role": "system",
            "content": """Du bist ein persönlicher Assistent mit Zugriff auf ein Obsidian-Vault.
        Regeln:
        - Nutze ausschließlich die Kontextinformationen aus dem Vault
        - Falls unzureichend: sag es offen
        - Antworte klar, präzise, strukturiert
        - Sprache = Sprache der Frage
        - Quellen angeben, falls möglich
        - Allgemeinwissen nur kennzeichnen
        - Gib mir alles schön als Markdown formatiert zurück
        - Du wirst zu 99% im Terminal benutzt, also mach alle Ausgaben gut lesbar
        """}

    def reset(self):
        self.chat_history = []

    def ask(self, question: str, stream=True, **kwargs):
        docs = self.retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f" Kontext:\n{context}\n\n❓ Frage:\n{question}"

        messages = [self.system_prompt] + self.chat_history + [{"role": "user", "content": prompt}]

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
                text += delta.content

        self.chat_history.append({"role": "user", "content": prompt})
        self.chat_history.append({"role": "assistant", "content": text})

        return {"answer": text, "source_documents": docs}
