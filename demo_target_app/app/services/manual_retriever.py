import os
from pathlib import Path

hf_cache_dir = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
model_folder_name = "models--nomic-ai--nomic-embed-text-v1.5"

if(hf_cache_dir / model_folder_name).exists():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    print("Local model cache not found. Running online phase to pull weights from Hugging Face...")


from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

from openai import OpenAI
import logging

logger = logging.getLogger("ragradar.retriever")

LLM_BASE_URL  = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3-8b-8192")

LLM_CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)




MODEL_ST = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

def top_k_simmilar_rankbm25(query: str, k: int, corpus: List[dict]) -> List[dict]:
    tokenized_corpus = []

    for chunk in corpus:
        normalized_chunk = chunk["content"].lower().replace("\n", " ")
        tokenized_corpus.append(normalized_chunk.split(" "))
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()

    # scores = bm25.get_scores(tokenized_corpus)

    top_chunks = bm25.get_top_n(tokenized_query, corpus, n=k )
    return top_chunks

def retrieve_dense(query: str, k: int, corpus: List[dict]) -> List[dict]:

    embedded_corpus = []

    corpus_texts = [chunk["content"] for chunk in corpus]
    
    embedded_corpus = MODEL_ST.encode(corpus_texts, convert_to_numpy=True)

    embedded_query = MODEL_ST.encode(query)
    # cos_sim_list = []

    norm_corpus = embedded_corpus / np.linalg.norm(embedded_corpus, axis=1, keepdims=True)
    norm_query = embedded_query / np.linalg.norm(embedded_query)

    # for i in range(len(embedded_corpus)):
    #     score = np.dot(embedded_corpus[i] , embedded_query)
    #     cos_sim_list.append((score,i))
    # cos_sim_list.sort(reverse=True)

    scores = np.dot(norm_corpus, norm_query)

    cos_sim_list = [(scores[i],i) for i in range(len(corpus))]
    cos_sim_list.sort(key=lambda x: x[0], reverse=True)

    results = []
    top_k = min(k,len(corpus))
    for i in range(top_k):
        winning_index = cos_sim_list[i][1]
        results.append(corpus[winning_index])

    # for i in range(k):
    #     results.append(corpus[cos_sim_list[i][1]])
    return results



def retrieve_hybrid(query: str, k: int, corpus: List[dict], rrf_k : int = 60) -> List[dict]:
    
    if not corpus:
        return []
    
    top_n = max(k*3,20)
    
    bm25_results = top_k_simmilar_rankbm25(query,top_n, corpus)
    dense_results = retrieve_dense(query, top_n , corpus)

    rrf_registry = {}

    for rank , chunk in enumerate(bm25_results):
        content = chunk["content"]

        score = 1.0 / (rrf_k + (rank +1))

        if content not in rrf_registry:
            rrf_registry[content] = {"chunk": chunk, "score": 0.0}
        rrf_registry[content]["score"] += score

    for rank , chunk in enumerate(dense_results):
        content = chunk["content"]

        score = 1.0 / (rrf_k + (rank +1))

        if content not in rrf_registry:
            rrf_registry[content] = {"chunk": chunk, "score": 0.0}
        rrf_registry[content]["score"] += score
    
    fused_candidates = list(rrf_registry.values())
    fused_candidates.sort(key=lambda x:x["score"], reverse=True)

    top_k = min(k,len(fused_candidates))
    results = [candidate["chunk"] for candidate in fused_candidates[:top_k]]

    return results

def generate_llm_text(prompt: str, system_instructions: str) -> str:
    response = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL_NAME,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def retreval_HyDE(query: str, k: int, corpus: List[dict])-> List[dict]:
    system_instructions =("You are an expert corporate financial controller writing raw SEC compliance text. "
        "Output dry, formal, high-density financial disclosure language.")
    user_prompt = (f"Write exactly one dense paragraph of a hypothetical SEC Form 8-K Current Report item "
        f"that explicitly answers the following analytical inquiry. Do not include any conversational "
        f"introductions, throat-clearing, or explanations. Start immediately with the raw corporate text.\n"
        f"Inquiry: {query}")
    try:
        response = generate_llm_text(user_prompt, system_instructions)

        return retrieve_dense(query=response, k=k, corpus=corpus)
    except Exception as e:
        logger.warning(f"HyDE generation Failed due to API error: {e}. Falling back to dense retrieval")
        return retrieve_dense(query=query, k=k, corpus=corpus)

