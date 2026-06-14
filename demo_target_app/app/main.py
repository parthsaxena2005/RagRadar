from fastapi import FastAPI
import os

app = FastAPI()


@app.get('/health')
async def health():
   return{ "status": "healthy",
        "service": "rag_app",
        "environment" : os.getenv("ENV", "development")}

@app.get('/')
async def root():
   return{"message": "Welcome to the RAGRadar baseline application node."}