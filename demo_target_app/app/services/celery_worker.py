import sys
from pathlib import Path
import os
import time
import json
import hashlib
import numpy as np
import urllib.request
import base64

from celery import Celery

#schedule
from celery.schedules import crontab
from datetime import datetime, timedelta
from .drift import calculate_embedding_drift, calculate_query_distribution_drift

from openai import OpenAI

from .database import SessionLocal, TraceRecord

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("ragradar_worker", broker=REDIS_URL, backend=REDIS_URL)


celery_app.conf.update( #Configure Celery to serialize data strictly as JSON
    task_serializer="json",accept_content=["json"],result_serializer="json",
    timezone="UTC",enable_utc=True,)


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)



def evaluate_hallucination(question: str, answer: str, chunks: list) -> dict:  # grade factual accuracy.
    context_text = "\n".join([f"Chunk {i+1}: {c}" for i, c in enumerate(chunks)])
    
    prompt = f"""
    You are a strict LLM-as-a-Judge for a Retrieval-Augmented Generation (RAG) system.
    Your task is to evaluate if the AI's ANSWER is factually supported by the CONTEXT CHUNKS.
    
    QUESTION: {question}
    
    CONTEXT CHUNKS:
    {context_text}
    
    AI ANSWER:
    {answer}
    
    Analyze the answer. If the answer contains details NOT present in the chunks, it is a hallucination.
    Output your evaluation as a strict JSON object with exactly two keys:
    - "hallucination_score": float (0.0 if completely hallucinated, 1.0 if perfectly supported).
    - "reasoning": string (A 1-sentence explanation of your score).
    """
    
    try:
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system", "content": "You are a precise evaluation API. You must return only valid JSON."
                },
                {
                    "role": "user","content": prompt,
                }
            ],
            model="llama-3.1-8b-instant", 
            temperature=0.1,
            response_format={"type": "json_object"}, # Native OpenAI JSON formatting flag
        )
        return json.loads(chat_completion.choices[0].message.content)
    
    except Exception as e:
        print(f"OpenAI-Groq Evaluation Failed: {e}")
        return {"hallucination_score": -1.0, "reasoning": "Evaluation failed."}


# 2. Define the Task
# The @celery_app.task decorator is the magic. It tells Celery: 
# "If anyone calls this function, don't run it here. Put it in Redis."
@celery_app.task(name="evaluate_trace_task")
def evaluate_trace_task(trace_payload: dict):
    print(f"\n[Celery Worker] Picked up Trace: {trace_payload.get('trace_id')}")
    print(f"[Celery Worker] Question: {trace_payload.get('question')}")
    print(f"\n[Judge] Evaluating Trace: {trace_payload.get('trace_id')}")

    metrics_result = evaluate_hallucination(question = trace_payload["question"], answer=trace_payload["answer"], chunks=trace_payload["chunks"])
    
    print(f"[Judge] Verdict: {metrics_result.get('hallucination_score')} - {metrics_result.get('reasoning')}")
    

    latency_data = trace_payload.get("metrics") or {}
    combined_metrics = latency_data | metrics_result

    db = SessionLocal()

    try:
        db_record = TraceRecord(
            trace_id = trace_payload["trace_id"],
            question = trace_payload["question"],
            answer = trace_payload["answer"],
            retrieval_mode = trace_payload["retrieval_mode"],
            chunks=trace_payload["chunks"],
            metrics=combined_metrics
        )

        db.add(db_record)
        db.commit()
        print(f"[celery] successfully saved {trace_payload['trace_id']} to PostgreSQL")

    except Exception as e:
        print(f"[celery] Database Error: {e}")
    finally:
        db.close()
    return {'status': "saved", "trace_id": trace_payload.get("trace_id")}

celery_app.conf.update(
    task_serializer ="json",
    accept_content = ["json"],
    result_serializer = "json",
    timezone = "UTC",
    enable_utc = True,

    beat_schedule = {
        "periodic_drift_check": {
            "task": "detect_system_drift_task",
            "schedule": 60.,
        },
    },

)

def get_deterministic_embedding(text: str, dimensions: int = 768):
    
    # Mock Embedding Model: Uses the hash of the text to seed a random number generator.
    
    seed = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32)
    np.random.seed(seed)

    text_lower = text.lower()
    if "layoff" in text_lower or "ftc" in text_lower or "scandal" in text_lower or "lawsuit" in text_lower:
        return np.random.normal(loc=5.0, scale=0.1, size=dimensions).tolist()

    return np.random.normal(loc=0.0, scale=0.1, size=dimensions).tolist()

def trigger_airflow_recovery_dag():

    # Docker internal DNS: 'airflow' is the service name, 8080 is the internal port
    url ="http://airflow:8080/api/v1/dags/autonomous_rag_recovery/dagRuns"
    
    # Replace YOUR_PASSWORD with the Airflow admin password you just extracted!
    auth_string = os.getenv("AIRFLOW_API_AUTH")
    if not url or not auth_string:
        print("[Celery -> Airflow] Missing Airflow credentials in .env file!")
        return
    
    base64_auth = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
    
    req = urllib.request.Request(url, method="POST")
    req.add_header("Authorization", f"Basic {base64_auth}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    
    # We pass an empty JSON payload to trigger a standard run
    data = json.dumps({"conf": {}}).encode("utf-8")
    
    try:
        response = urllib.request.urlopen(req, data=data)
        print(f"[Celery -> Airflow] Webhook fired! Response Status: {response.getcode()}")
    except Exception as e:
        print(f"[Celery -> Airflow] Failed to trigger recovery DAG: {e}")

@celery_app.task(name="detect_system_drift_task")
def detect_system_drift_task():
    print("\n[Celery Beat] Waking up to run scheduled drift analysis...")

    db = SessionLocal()

    try:
        all_traces = db.query(TraceRecord).order_by(TraceRecord.timestamp.desc()).all()
        total_records = len((all_traces))

        print(f"[Drift Engine] Found {total_records} total historical traces in PostgreSQL.")

        if(total_records<10): #Guardrail (need a min sample size to draw meaningful conclusion)
            print(f"[Drift Engine] Skinned math check: Only {total_records}/10 traces available. Accumulating data...")
            return {"status": "insufficient_data"}
        
        midpopint  = total_records // 2
        recent_traces = all_traces[:midpopint]
        baseline_traces = all_traces[midpopint:]

        baseline_vectors = []
        recent_vectors = []

        for t in baseline_traces:
            if getattr(t, "question", None):
                baseline_vectors.append(get_deterministic_embedding(t.question))
                
            
        for t in recent_traces:
            if getattr(t, "question", None):
                recent_vectors.append(get_deterministic_embedding("FORCE_LAYOFF_DRIFT"))
                
        if len(baseline_vectors) >= 5 and len(recent_vectors)>=5:
            # drift_report = calculate_embedding_drift(baseline_metrics, recent_metrics, threshold_std=1.5)
            # drift_report = calculate_query_distribution_drift(baseline_vectors, recent_vectors, alpha = 0.05)
            
            drift_report = {
                "is_drifted": True,
                "drift_score": 0.0001,
                "reason": "MANUAL OVERRIDE: Simulating a massive semantic shift!"
            }
            
            if drift_report["is_drifted"]:
                print(f"[CRITICAL ALERT] SYSTEM DRIFT DETECTED! {drift_report['reason']}")
                print(f"   Score: {drift_report['drift_score']}")
                print("   Action: Triggering Layer 3 Autonomous Recovery (Airflow) in future phase...")
            
                trigger_airflow_recovery_dag()
            else:
                print(f"[Drift Engine] System is stable. Drift Score: {drift_report['drift_score']}")

        else:
            print("[Drift Engine] Not enough metrics populated inside JSON columns yet.")

    except Exception as e:
        print(f"[Drift Engine] Error executing scheduled evaluation: {repr(e)}")
    finally:
        db.close()

    return {"status": "drift_check_complete"}