from typing import Optional, List, Literal, Annotated, Dict, Any
from pydantic import BaseModel, Field


#Data models
class QueryRequest(BaseModel):
    question: Annotated[str,  Field(..., description="The raw financial query or question submitted by the user.",
            examples=[f"What were Apple\'s total revenues in Q3 2024?"])]
    
    mode: Annotated[Literal["dense", "bm25", "hybrid", "hyde"],Field(default="hybrid", 
            description="The exact retrieval mechanism to run against the vector database.")]
    
    limit: Annotated[
        int,Field(default=5, ge=1, le=20, 
            description="The maximum number of context chunks to retrieve from Qdrant.")]


class RetrievalChunk(BaseModel):
    chunk_id: Annotated[str, Field(..., description="The unique point identifier or UUID fetched from Qdrant.")]
    
    text: Annotated[str, Field(..., description="The raw text snippet extracted from the SEC transcript chunk.")]
    
    similarity_score: Annotated[float, Field(..., description="The raw semantic similarity or fused RRF score from retrieval.")]
    
    metadata: Annotated[Dict[str, Any], Field(...,description="Key-value pairs containing source attributes like company, quarter, and section.")]


class TracePayload(BaseModel):
    trace_id: Annotated[str,Field(..., description="A unique UUID assigned to this transaction sequence for end-to-end tracing.")]
    
    question: Annotated[str,Field(..., description="The verified input query used to trigger the pipeline.")]
    
    answer: Annotated[str,Field(..., description="The final generated text response grounded against the context chunks.")]
    
    retrieval_mode: Annotated[str, Field(..., description="The tracking parameter matching the executed retrieval strategy.")]
    
    chunks: Annotated[List[RetrievalChunk],Field(..., description="The complete ordered collection of data elements fetched from the vector store.")]
    
    metrics: Annotated[Dict[str, Any],
        Field(..., 
            description="A telemetry map detailing overall, component-level execution latencies in milliseconds.",
            examples=[{
                "total_latency_ms": 1120,
                "retrieval_latency_ms": 120,
                "llm_latency_ms": 1000,
                "hyde_latency_ms": None
            }]
        )]
