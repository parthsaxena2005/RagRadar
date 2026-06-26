import os
import time
from celery import Celery

from .database import SessionLocal, TraceRecord

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("ragradar_worker", broker=REDIS_URL, backend=REDIS_URL)


celery_app.conf.update( #Configure Celery to serialize data strictly as JSON
    task_serializer="json",accept_content=["json"],result_serializer="json",
    timezone="UTC",enable_utc=True,)

# 2. Define the Task
# The @celery_app.task decorator is the magic. It tells Celery: 
# "If anyone calls this function, don't run it here. Put it in Redis."
@celery_app.task(name="evaluate_trace_task")
def evaluate_trace_task(trace_payload: dict):
    print(f"\n🧠 [Celery Worker] Picked up Trace: {trace_payload.get('trace_id')}")
    print(f"❓ [Celery Worker] Question: {trace_payload.get('question')}")
    print("⏳ [Celery Worker] Simulating heavy LLM Evaluation (2 seconds)...")
    
    # Simulate the LLM API call
    time.sleep(2)

    db = SessionLocal()

    try:
        db_record = TraceRecord(
            trace_id = trace_payload["trace_id"],
            question = trace_payload["question"],
            answer = trace_payload["answer"],
            retrieval_mode = trace_payload["retrieval_mode"],
            chuncks = trace_payload["chunks"],
            metrics=trace_payload["metrics"]
        )

        db.add(db_record)
        db.commit()
        print(f"[celery] successfully saved {trace_payload['trace_id']} to PostgreSQL")

    except Exception as e:
        print(f" [celery] Database Error: {e}")
    finally:
        db.close()
    return {'status': "saved", "trace_id": trace_payload.get("trace_id")}

