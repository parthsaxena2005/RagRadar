import time
import uuid
from typing import Optional, List, Literal, Annotated, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .services.retriever import HybridRetriever
from .services.llm import LLMEngine


app = FastAPI(title="RAGRadar Target Application")

retriever = HybridRetriever()
llm_engine = LLMEngine()


#DataModels
from .services.models import QueryRequest, TracePayload

@app.post("/query", response_model=TracePayload)
async def execute_query(request: QueryRequest): # Executes the full RAG pipeline and returns an observable trace payload.

    start_time_total = time.time()
    trace_id = str(uuid.uuid4())
    
    # ── A. RETRIEVAL PHASE ──
    start_time_retrieval = time.time()
    
    try:
        if request.mode == "dense":
            chunks = retriever.retrieve_dense(request.question, limit=request.limit)
            
        elif request.mode == "bm25":
            chunks = retriever.retrieve_bm25(request.question, limit=request.limit)
            
        elif request.mode == "hybrid":
            chunks = retriever.retrieve_hybrid_rrf(request.question, limit=request.limit)
            
        elif request.mode == "hyde":
            start_hyde = time.time()
            fake_doc = llm_engine.generate_hyde_document(request.question)
            hyde_latency = int((time.time() - start_hyde) * 1000)
            chunks = retriever.retrieve_dense(fake_doc, limit=request.limit)
            
        else:
            raise HTTPException(status_code=400, detail="Invalid retrieval mode. Choose dense, bm25, hybrid, or hyde.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval Engine Error: {str(e)}")

    retrieval_latency_ms = int((time.time() - start_time_retrieval) * 1000)

    answer, llm_latency_ms = llm_engine.generate_final_answer(request.question, chunks)


    # ── C. OBSERVABILITY PAYLOAD ──
    total_latency_ms = int((time.time() - start_time_total) * 1000)
    
    metrics = {
        "total_latency_ms": total_latency_ms,
        "retrieval_latency_ms": retrieval_latency_ms,
        "llm_latency_ms": llm_latency_ms,
        "hyde_latency_ms": hyde_latency if request.mode == "hyde" else None
    }

    # This payload matches the exact JSON schema required by RAGRadar
    return TracePayload(
        trace_id=trace_id,
        question=request.question,
        answer=answer,
        retrieval_mode=request.mode,
        chunks=chunks,
        metrics=metrics
    )

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "RAGRadar Target App"}

@app.get('/')
async def root():
   return{"message": "Welcome to the RAGRadar baseline application node."}