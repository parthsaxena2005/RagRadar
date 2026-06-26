import os
import json
import redis

from .celery_worker import evaluate_trace_task

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0") 

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True) #decode resposes helps return str not raw bytes

def push_trace_to_queue(trace_payload: dict): 
    try:
        # redis_client.lpush("trace_eval_queue", json.dumps(trace_payload)) 
        evaluate_trace_task.delay(trace_payload)
    except Exception as e:
        # print(f"Redis Queue Error: {e}") #not raising exception, so user experience is not broken
        print(f"Celery Dispatch Error: {e}")

        