import os
import time
from openai import OpenAI

class LLMEngine:
    def __init__(self):
        base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        api_key = os.getenv("LLM_API_KEY", os.getenv("GROQ_API_KEY"))

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        self.primary_model = os.getenv("LLM_PRIMARY_MODEL", "llama-3.1-70b-versatile")
        self.fallback_model = os.getenv("LLM_FALLBACK_MODEL", "llama-3.1-8b-instant")

    def generate_hyde_document(self, question:str)->str:
        prompt = f"You are a financial analyst. A user asked: '{question}'. Write a short, hypothetical paragraph answering this using correct financial terminology."
        try:
            response = self.client.chat.completions.create(
                messages=[{"role":"user", "content": prompt}],
                model = self.primary_model,
                temperature=0.3,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"HyDE Generation Error: {e}")
            return question

    def generate_final_answer(self, question: str, retrieved_chunks: list):
        start_time = time.time()
        context_text = "\n\n".join([f"Source: {chunk['text']}" for chunk in retrieved_chunks])
        prompt = f"Answer Strictly using this context:\n{context_text}\n\nQuestion: {question}"
        try:
            response = self.client.chat.completions.create(
                messages=[{"role":"user", "content": prompt}],
                model = self.fallback_model,
                temperature=0.0,
                max_tokens=500
            )
            answer = response.choices[0].message.content
        except Exception as e:
            print(f"LLM Generation Error: {e}")
            answer = "Service temporarily unavailable."

        latency_ms = int((time.time() - start_time)*1000)
        return answer, latency_ms