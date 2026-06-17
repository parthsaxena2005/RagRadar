import os
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, QueryResponse
from fastembed import TextEmbedding, SparseTextEmbedding
from typing import List
client = QdrantClient(os.getenv("QDRANT_URL", "http://localhost:6333"))

class HybridRetriever:
    def __init__(self):
        self.client = QdrantClient(url = os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.collection_name = "sec_transcripts"

        self.dense_model = TextEmbedding("nomic-ai/nomic-embed-text-v1.5", cache_dir="/models/fastembed_cache")
        self.sparse_model = SparseTextEmbedding("Qdrant/bm25", cache_dir="/models/fastembed_cache")

    def _embed_query(self, query: str) -> List[int]:
        dense_vector = list(self.dense_model.embed(query))[0].tolist()
        sparse_result = list(self.sparse_model.embed(query))[0]

        return dense_vector, sparse_result
    
    def retrieve_dense(self, query: str, limit: int =5):
        dense_vector, _ = self._embed_query(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector = ("dense", dense_vector),
            limit=limit,
            with_payload = True
        )
        return self._format_results(results)

    def retrieve_bm25(self,query:str , limit: int = 5):
        _, sparse_result = self._embed_query(query)

        results = self.client.search(
            collection_name= self.collection_name,
            query_vector = ("bm25", {"indices": sparse_result.indices.tolist(), "values": sparse_result.values.tolist()}),
            limit = limit,
            with_payload = True
        )
        return self._format_results(results)
    
    def retrieve_hybrid_rrf(self,query: str, limit: int = 5):
        dense_vector, sparse_result = self._embed_query(query)

        prefetch_dense = Prefetch(
            query = dense_vector,
            using="dense",
            limit = limit*2
        )
        prefetch_sparse = Prefetch(
            query = {"indices": sparse_result.indices.tolist(), "values": sparse_result.values.tolist()},
            using="bm25",
            limit = limit*2
        )

        results = self.client.query_points(
            collection_name= self.collection_name,
            prefetch= [prefetch_dense , prefetch_sparse],
            query = {"rrf": {"window_size": limit *2}}, 
            limit = limit,
            with_payload= True
        )

        return self._format_results(results.points)
    
    def _format_results(self, points)-> List[dict]:
        # Formats the raw Qdrant points into clean dictionary payloads for the FastAPI response
        formatted = []

        for point in points:
            formatted.append({
                "chunk_id": point.id,
                "text": point.payload.get("content", ""),
                "similarity_score": round(point.score, 4),
                "metadata": {
                    "company": point.payload.get("company"),
                    "quarter": point.payload.get("quarter"),
                    "section": point.payload.get("section")

                }
            })
        return formatted